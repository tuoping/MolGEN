from .ema import ExponentialMovingAverage
from .logger import get_logger

logger = get_logger(__name__)

import torch, time
from torch import nn
import copy
import numpy as np
from functools import partial

from .model.equivariant_latent_model import EquivariantTransformer_dpm, Encoder_dpm, Processor, Decoder
from .wrapper import Wrapper, gather_log, get_log_mean


from pymatgen.core import Molecule
from pymatgen.analysis.molecule_matcher import BruteForceOrderMatcher, GeneticOrderMatcher, HungarianOrderMatcher, KabschMatcher
from pymatgen.io.xyz import XYZ

# Typing
from torch import Tensor
from typing import List, Optional, Tuple
from .transport.transport import create_transport, Sampler

_TORCH_FLOAT_PRECISION=torch.float32

map_to_chemical_symbol = {
    0: "H",
    1: 'C',
    2: "N",
    3: "O"

}

def xh2pmg(species, xh):
    mol = Molecule(
        species=species,
        coords=xh[:, :3].cpu().numpy(),
    )
    return mol


def xyz2pmg(xyzfile):
    xyz_converter = XYZ(mol=None)
    mol = xyz_converter.from_file(xyzfile).molecule
    return mol


def rmsd_core(mol1, mol2, threshold=0.5, same_order=False):
    _, count = np.unique(mol1.atomic_numbers, return_counts=True)
    if same_order:
        bfm = KabschMatcher(mol1)
        _, rmsd = bfm.fit(mol2)

        # Raw-centered RMSD (translation removed, no rotation)
        A = np.asarray(mol1.cart_coords, dtype=np.float64)
        B = np.asarray(mol2.cart_coords, dtype=np.float64)
        A0 = A - A.mean(0, keepdims=True)
        B0 = B - B.mean(0, keepdims=True)
        rmsd_raw_centered = float(np.sqrt(((A0 - B0) ** 2).sum(axis=1).mean()))
        if rmsd_raw_centered < rmsd:
            print(mol1.species, mol2.species)
            print(mol1.cart_coords, mol2.cart_coords)
            raise RuntimeError

        return rmsd
    total_permutations = 1
    for c in count:
        total_permutations *= np.math.factorial(c)  # type: ignore
    if total_permutations < 1e4:
        bfm = BruteForceOrderMatcher(mol1)
        _, rmsd = bfm.fit(mol2)
    else:
        bfm = GeneticOrderMatcher(mol1, threshold=threshold)
        pairs = bfm.fit(mol2)
        rmsd = threshold
        for pair in pairs:
            rmsd = min(rmsd, pair[-1])
        if not len(pairs):
            bfm = HungarianOrderMatcher(mol1)
            _, rmsd = bfm.fit(mol2)
    return rmsd


def pymatgen_rmsd(
    species, 
    mol1,
    mol2,
    ignore_chirality: bool = False,
    threshold: float = 0.5,
    same_order: bool = True,
):
    if isinstance(mol1, str):
        mol1 = xyz2pmg(species, mol1)
    if isinstance(mol2, str):
        mol2 = xyz2pmg(species, mol2)
    rmsd = rmsd_core(mol1, mol2, threshold, same_order=same_order)
    if ignore_chirality:
        coords = mol2.cart_coords
        coords[:, -1] = -coords[:, -1]
        mol2_reflect = Molecule(
            species=mol2.species,
            coords=coords,
        )
        rmsd_reflect = rmsd_core(
            mol1, mol2_reflect, threshold, same_order=same_order)
        rmsd = min(rmsd, rmsd_reflect)
    return rmsd

def batch_rmsd_sb(
    species: List[str],
    fragments_node,
    pred_xh: Tensor,
    target_xh: Tensor,
    threshold: float = 0.5,
    same_order: bool = True,
) -> List[float]:

    rmsds = []
    end_ind = np.cumsum(fragments_node.long().cpu().numpy())
    start_ind = np.concatenate([np.int64(np.zeros(1)), end_ind[:-1]])
    for start, end in zip(start_ind, end_ind):
        mol1 = xh2pmg(species[start:end], pred_xh[start : end])
        mol2 = xh2pmg(species[start:end], target_xh[start : end])
        rmsd = pymatgen_rmsd(
            species[start:end], 
            mol1,
            mol2,
            ignore_chirality=True,
            threshold=threshold,
            same_order=same_order,
        )
        rmsds.append(min(rmsd, 1.0))
    return rmsds


class EquivariantMDGenWrapper(Wrapper):
    def __init__(self, args):
        super().__init__(args)
        for key in [
            'cond_interval',
        ]:
            if not hasattr(args, key):
                setattr(args, key, False)
        
        num_species = args.num_species
        num_radial = 96
        if args.design:
            num_scalar_out = self.args.num_species
            num_vector_out=0
        else:
            num_scalar_out = 0
            num_vector_out=1
        latent_dim = args.embed_dim
        
        if args.tps_condition:
            encoder = Encoder_dpm(num_species, latent_dim, (64+48+8)*3, latent_dim, input_dim=1, cv_dim=1, object_aware=args.object_aware)
        elif args.sim_condition:
            encoder = Encoder_dpm(num_species, latent_dim, (64+48+8)*2, latent_dim, input_dim=1, cv_dim=1, object_aware=args.object_aware)
        else:
            encoder = Encoder_dpm(num_species, latent_dim, (64+48+8), latent_dim, input_dim=1, cv_dim=1, object_aware=args.object_aware)

        processor = Processor(num_convs=5, node_dim=latent_dim, num_heads=8, ff_dim=args.ff_dim, edge_dim=latent_dim)
        print("Initializing drift model")
        self.model = EquivariantTransformer_dpm(
            encoder = encoder,
            processor = processor,
            decoder = Decoder(dim=latent_dim, num_scalar_out=num_scalar_out, num_vector_out=num_vector_out, num_species=args.num_species),
            cutoff=args.cutoff,
            latent_dim=latent_dim,
            num_radial = num_radial,
            design=args.design,
            potential_model = False,
            tps_condition=args.tps_condition,
            sim_condition=args.sim_condition,
            num_species=args.num_species,
            pbc=args.pbc,
            object_aware=args.object_aware,
        )
        if args.potential_model:
            num_scalar_out = 1
            num_vector_out = 0
            self.potential_model = EquivariantTransformer_dpm(
                encoder = encoder,
                processor = processor,
                decoder = Decoder(dim=latent_dim, num_scalar_out=num_scalar_out, num_vector_out=num_vector_out, num_species=args.num_species),
                cutoff=args.cutoff,
                latent_dim=latent_dim,
                design=args.design,
                potential_model = args.potential_model,
                tps_condition=args.tps_condition,
                sim_condition=args.sim_condition,
                num_species=args.num_species,
                pbc=args.pbc,
                object_aware=args.object_aware
            )
        if args.path_type == "Schrodinger_Linear":
            print("Initializing score model")
            self.score_model = EquivariantTransformer_dpm(
                encoder = encoder,
                processor = processor,
                decoder = Decoder(dim=latent_dim, num_scalar_out=num_scalar_out, num_vector_out=num_vector_out),
                cutoff=args.cutoff,
                latent_dim=latent_dim,
                design=args.design,
                potential_model = args.potential_model,
                tps_condition=args.tps_condition,
                sim_condition=args.sim_condition,
                pbc=args.pbc,
                object_aware=args.object_aware,
            )
        else:
            self.score_model = None

        self.transport = create_transport(
            args,
            args.path_type,
            args.prediction,
            train_eps=1e-5,
            sample_eps=1e-5,
            score_model=self.score_model
        )
        self.transport_sampler = Sampler(self.transport)

        if not hasattr(args, 'ema'):
            args.ema = False
        if args.ema:
            self.ema = ExponentialMovingAverage(
                model=self.model, decay=args.ema_decay
            )
            self.cached_weights = None

        if self.args.precision == '32-true':
            _TORCH_FLOAT_PRECISION = torch.float32

    def on_validation_epoch_end(self):
        if self.args.ema:
            self.restore_cached_weights()
        log = self._log
        log = {key: log[key] for key in log if "val_" in key}
        log = gather_log(log, self.trainer.world_size)
        mean_log = get_log_mean(log)
        self.log("val_loss", mean_log['val_loss'])
        # self.log("val_loss_gen", mean_log['val_loss_gen'])
        # self.log("val_meanRMSD_Kabsch", mean_log['val_meanRMSD_Kabsch'])
        self.print_log(prefix='val', save=False)

    def prep_batch(self, batch):
        if self.args.design:
            return self.prep_batch_species(batch)
        else:
            return self.prep_batch_x(batch)

    def prep_batch_species(self, batch):
        species = batch["species"]
        latents = batch["species"]
        x_now = batch["x"]
        
    
        B, T, L, num_elem = species.shape

        
        if self.args.design:
            loss_mask = batch["mask"]
            # loss_mask = torch.cat([h_loss_mask, v_loss_mask], -1)
            loss_mask = loss_mask
        else:
            v_loss_mask = batch["v_mask"]
            loss_mask = v_loss_mask


        B, T, L, _ = latents.shape
        assert _ == self.args.num_species, f"latents shape should be (B, T, D, self.args.num_species), but got {latents.shape}"
        ########
        cond_mask = torch.zeros(B, T, L, dtype=int, device=species.device)
        if self.args.sim_condition:
            cond_mask[:, 0] = 1
        if self.args.cond_interval:
            cond_mask[:, ::self.args.cond_interval] = 1
        return {
            "species": latents,
            "latents": latents,
            'loss_mask': loss_mask,
            'model_kwargs': {
                "cell": batch["cell"],
                "num_atoms": batch["num_atoms"],
                "conditions": None,
                "aatype": None,
                "x_latt": x_now,
            }
        }

    def prep_batch_x(self, batch):
        species = batch["species"]
        latents = batch["x"]
        # rdf = batch["RDF"]
        B, T, L, num_elem = species.shape

        v_loss_mask = batch["v_mask"]


        B, T, L, _ = latents.shape
        assert _ == 3, f"latents shape should be (B, T, D, 3), but got {latents.shape}"
        ########
        
        if "TKS_mask" not in batch.keys():
            batch['TKS_mask'] = torch.ones(B,T,L, dtype=int, device=species.device)
            batch['TKS_v_mask'] = torch.ones(B,T,L,3, dtype=int, device=species.device)

        if self.args.sim_condition:
            cond_mask_f = torch.zeros(B, T, L, dtype=int, device=species.device)
            cond_mask = torch.zeros(B, T, L, dtype=int, device=species.device)
            cond_mask_f[:, 0] = 1
            cond_mask[:, -1] = 1
            if self.stage == "inference":
                conditional_batch = True
            else:
                conditional_batch = torch.rand(1)[0] >= 1-self.args.ratio_conditonal
                # conditional_batch = True

        elif self.args.tps_condition:
            cond_mask_f = torch.zeros(B, T, L, dtype=int, device=species.device)
            cond_mask_r = torch.zeros(B, T, L, dtype=int, device=species.device)
            cond_mask = torch.zeros(B, T, L, dtype=int, device=species.device)
            cond_mask_f[:, 0] = 1
            cond_mask_r[:, -1] = 1
            cond_mask[:, 1:-1] = 1
            if self.stage == "inference":
                conditional_batch = True
            else:
                conditional_batch = torch.rand(1)[0] >= 1-self.args.ratio_conditonal
                # conditional_batch = True

        else:
            conditional_batch = None
        if (self.args.tps_condition and conditional_batch):
            return {
                "species": species.to(_TORCH_FLOAT_PRECISION),
                "latents": latents.to(_TORCH_FLOAT_PRECISION),
                'E': batch['e_now'].to(_TORCH_FLOAT_PRECISION),
                'loss_mask': batch["TKS_v_mask"]*cond_mask.unsqueeze(-1).to(_TORCH_FLOAT_PRECISION),
                'loss_mask_potential_model': (batch["TKS_mask"]!=0).to(int)[:,:,0]*cond_mask[:,:,0],
                'model_kwargs': {
                    "x1": latents.to(_TORCH_FLOAT_PRECISION),
                    'v_mask': (batch["TKS_v_mask"]!=0).to(int)*cond_mask.unsqueeze(-1),
                    "aatype": species.to(_TORCH_FLOAT_PRECISION),
                    "cell": batch['cell'].to(_TORCH_FLOAT_PRECISION),
                    "num_atoms": batch["num_atoms"],
                    "fragments_idx": batch['fragments_idx'],
                    "conditions": {
                        'cond_f':{
                            'x': latents[:,0,...].unsqueeze(1).expand(B,T,L,3).reshape(-1,3).to(_TORCH_FLOAT_PRECISION),             # Only using the 1st configuration as cond_f
                            "fragments_idx": batch['fragments_idx'][:,0,...].unsqueeze(1).expand(B,T,L).reshape(-1),
                            'mask': cond_mask_f[:,0,...].unsqueeze(1).expand(B,T,L).reshape(-1),          # Since only 1st configuration is inputed and cond_mask already masked the prediction only to the TPS, cond_mask_f here is a place_holder
                        },
                        'cond_r':{
                            'x': latents[:,-1,...].unsqueeze(1).expand(B,T,L,3).reshape(-1,3).to(_TORCH_FLOAT_PRECISION),
                            "fragments_idx": batch['fragments_idx'][:,-1,...].unsqueeze(1).expand(B,T,L).reshape(-1),
                            'mask': cond_mask_r[:,-1,...].unsqueeze(1).expand(B,T,L).reshape(-1),
                        }
                    }
                },
                'conditional_batch': conditional_batch
            }
        elif (self.args.sim_condition and conditional_batch):
            return {
                "species": species.to(_TORCH_FLOAT_PRECISION),
                "latents": latents.to(_TORCH_FLOAT_PRECISION),
                # 'E': batch['e_now'].to(_TORCH_FLOAT_PRECISION),
                'loss_mask': batch["TKS_v_mask"]*cond_mask.unsqueeze(-1).to(_TORCH_FLOAT_PRECISION),
                'loss_mask_potential_model': (batch["TKS_mask"]!=0).to(int)[:,:,0]*cond_mask[:,:,0],
                'model_kwargs': {
                    "x1": latents[:,-1,...].unsqueeze(1).expand(B,T,L,3).to(_TORCH_FLOAT_PRECISION),
                    'v_mask': (batch["TKS_v_mask"]!=0).to(int)*cond_mask.unsqueeze(-1),
                    "aatype": species.to(_TORCH_FLOAT_PRECISION),
                    "cell": batch['cell'].to(_TORCH_FLOAT_PRECISION),
                    "num_atoms": batch["num_atoms"],
                    "fragments_idx": batch['fragments_idx'],
                    "conditions": {
                        'cond_f':{
                            'x': latents[:,0,...].unsqueeze(1).expand(B,T,L,3).reshape(-1,3).to(_TORCH_FLOAT_PRECISION),             # Only using the 1st configuration as cond_f
                            "fragments_idx": batch['fragments_idx'][:,0,...].unsqueeze(1).expand(B,T,L).reshape(-1),
                            'mask': cond_mask_f[:,0,...].unsqueeze(1).expand(B,T,L).reshape(-1),          # Since only 1st configuration is inputed and cond_mask already masked the prediction only to the TPS, cond_mask_f here is a place_holder
                        },
                        # 'cond_r':{
                        #     'x': latents[:,-1,...].unsqueeze(1).expand(B,T,L,3).reshape(-1,3).to(_TORCH_FLOAT_PRECISION),
                        #     "fragments_idx": batch['fragments_idx'][:,-1,...].unsqueeze(1).expand(B,T,L).reshape(-1),
                        #     'mask': cond_mask_r[:,-1,...].unsqueeze(1).expand(B,T,L).reshape(-1),
                        # }
                    }
                },
                'conditional_batch': conditional_batch
            }
        else:
            return {
                "species": species.to(_TORCH_FLOAT_PRECISION),
                "latents": latents.to(_TORCH_FLOAT_PRECISION),
                'loss_mask': v_loss_mask.to(_TORCH_FLOAT_PRECISION),
                'model_kwargs': {
                    "cv": batch['cv'].to(_TORCH_FLOAT_PRECISION),
                    "aatype": species.to(_TORCH_FLOAT_PRECISION),
                    'x1': latents.to(_TORCH_FLOAT_PRECISION),
                    'v_mask': (v_loss_mask!=0).to(int),
                    "cell": batch['cell'].to(_TORCH_FLOAT_PRECISION),
                    "num_atoms": batch["num_atoms"],
                    "conditions": None
                },
                'conditional_batch': conditional_batch
            }
    
    def general_step(self, batch, stage='train'):
        self.iter_step += 1
        self.stage = stage
        start1 = time.time()
        prep = self.prep_batch(batch)

        start = time.time()

        out_dict = self.transport.training_losses(
            model=self.model,
            x1=prep['latents'],
            aatype1=batch['species'],
            mask=prep['loss_mask'],
            model_kwargs=prep['model_kwargs'],
            global_step = self.current_epoch
        )
        self.prefix_log('model_dur', time.time() - start)
        self.prefix_log('time', out_dict['t'])
        # self.prefix_log('conditional_batch', prep['conditional_batch'].to(torch.float32))
        loss_gen = out_dict['loss']
        assert self.args.weight_loss_var_x0 == 0
        loss = loss_gen
        if self.score_model is not None:
            self.prefix_log("loss_flow", out_dict['loss_flow'])
            self.prefix_log("loss_score", out_dict['loss_score'])
        if self.args.KL == 'symm':
            self.prefix_log('loss_symmkl', out_dict['loss_symmkl'])
            # self.prefix_log('loss_entropy', out_dict['loss_entropy'])
            self.prefix_log('loss_l1', out_dict['loss_l1'])
        if self.args.KL == 'alpha':
            self.prefix_log('loss_alphadiv', out_dict['loss_alphadiv'])
            self.prefix_log('loss_l1', out_dict['loss_l1'])

        if self.args.potential_model:
            self.prefix_log('loss_gen', loss_gen)
            B,T,L,_ = prep["latents"].shape
            t = torch.ones((B,), device=prep["latents"].device).to(_TORCH_FLOAT_PRECISION)
            energy = self.potential_model(prep['latents'], t, **prep["model_kwargs"])
            energy = energy.sum(dim=2).squeeze(-1)
            # forces = -torch.autograd.grad(energy, prep['latents'])[0]
            loss_energy = (((energy -prep["E"])**2)*prep['loss_mask_potential_model']).sum(-1)
            self.prefix_log('loss_energy', loss_energy)        
            loss += loss_energy * 0.1

        self.prefix_log('model_dur', time.time() - start)
        self.prefix_log('loss', loss)

        self.prefix_log('dur', time.time() - self.last_log_time)
        if 'name' in batch:
            self.prefix_log('name', ','.join(batch['name']))
        self.prefix_log('general_step_dur', time.time() - start1)
        self.last_log_time = time.time()
        if stage == "val":
            # self._val_saddle_point_object_aware(batch, prep)
            pass

        if not torch.isfinite(loss.mean()):
            return None
        if torch.isnan(loss.mean()):
            return None
        return loss.mean()

    def _val_saddle_point_object_aware(self, batch, prep, stage="val"):
            B,T,L,_ = prep['latents'].shape
            try:
                pred_pos, _ = self.inference(batch, stage=stage)
                ref_pos = prep['latents']
                with torch.no_grad():
                    ## (\Delta d per atom) # B,T,L
                    err = ((((pred_pos - ref_pos)*(prep['loss_mask']!=0)).norm(dim=-1)))
                    ## RMSD per configuration # B,T
                    err = ((err**2).mean(dim=-1)).sqrt()
                    ## mean RMSD per sample # B
                    err = err.mean(dim=-1)
                    assert torch.all((prep['loss_mask']!=0)[:,0] == 0)
                    assert torch.all((prep['loss_mask']!=0)[:,-1] == 0)
                    assert torch.all((prep['loss_mask']!=0)[:,1] == 1)
                    assert T == 3
                    self.prefix_log('meanRMSD', err*3)  # An extra factor of 3 was divided when taking the mean over the T dimension

                with torch.no_grad():
                    assert torch.all((prep['loss_mask']!=0)[:,0] == 0)
                    assert torch.all((prep['loss_mask']!=0)[:,-1] == 0)
                    assert torch.all((prep['loss_mask']!=0)[:,1] == 1)
                    assert T == 3
                    labels = torch.argmax(prep["species"][:,1,...], dim=-1).ravel().cpu().numpy()  # B,T,L
                    symbols = [map_to_chemical_symbol[labels[i_elem]] for i_elem in range(len(labels))]
                    # fragments_node = torch.unique_consecutive(prep['model_kwargs']['fragments_idx'][:,1,...], return_counts=True)[1] # prep['model_kwargs']['num_atoms'][:,1].ravel() # reshape B,1 to B*1
                    fragments_node = prep['model_kwargs']['num_atoms'][:,1].ravel() # reshape B,1 to B*1
                    pred_xh = pred_pos[:,1,...].reshape(-1, 3) # reshape B,1,L,3 to B*1*L*3
                    target_xh = ref_pos[:,1,...].reshape(-1, 3) # reshape B,1,L,3 to B*1*L*3
                    try:
                        rmsds = batch_rmsd_sb(
                            symbols, fragments_node, pred_xh, target_xh, same_order = False)
                        self.prefix_log('meanRMSD_Kabsch', torch.tensor(rmsds).mean())
                    except:
                        self.prefix_log('meanRMSD_Kabsch', torch.nan)
            except:
                print("WARNNING:: Inference failed !!!")
                self.prefix_log('meanRMSD_Kabsch', torch.nan)

    def guided_velocity(self, x, t, cell=None, 
                num_atoms=None,
                conditions=None,
                aatype=None, x1=None, v_mask=None):
        with torch.no_grad(): 
            v = self.model.forward_inference(x, t,                 
                cell=cell, 
                num_atoms=num_atoms,
                conditions=conditions,
                aatype=aatype, x1=x1, v_mask=v_mask)
        B,T,L,_ = x.shape
        with torch.enable_grad():
            x = x.detach().requires_grad_(True)
            g = -torch.autograd.grad(self.potential_model(x, torch.ones((B,), device=x.device).detach().requires_grad_(False), 
                    cell=cell, 
                    num_atoms=num_atoms,
                    conditions=conditions,
                    aatype=aatype, x1=x1, v_mask=v_mask).sum(dim=2).squeeze(-1)[:,1], x, create_graph=False)[0].detach()
        self.integration_step += 1
        return v + self.args.guidance_pref*g

    
    def inference(self, batch, stage='inference'):
        s_time= time.time()
        self.stage = stage
        prep = self.prep_batch(batch)

        latents = prep['latents']
        B, T, N, D = latents.shape

        if self.args.design:
            # zs_continuous = torch.randn(B, T, N, self.latent_dim - self.args.num_species, device=latents.device)
            zs_discrete = torch.distributions.Dirichlet(torch.ones(B, N, self.args.num_species, device=latents.device)).sample()
            zs_discrete = zs_discrete[:, None].expand(-1, T, -1, -1)
            # zs = torch.cat([zs_continuous, zs_discrete], -1)
            zs = zs_discrete

            x1 = prep['latents']
            x_d = torch.zeros(x1.shape[0], x1.shape[1], x1.shape[2], self.args.num_species, device=self.device)
            xt = torch.cat([x1, x_d], dim=-1)
            logits = self.model.forward_inference(xt, torch.ones(B, device=self.device),
                                                  **prep['model_kwargs'])
            aa_out = torch.argmax(logits, -1)
            # aa_out = logits
            vector_out = prep["model_kwargs"]["x_latt"]
            return vector_out, aa_out
        else:
            zs = torch.randn(B, T, N, D, device=self.device)*self.args.x0std

        self.integration_step = 0
        if self.score_model is None:
            if self.args.likelihood:
                sample_fn = self.transport_sampler.sample_ode_likelihood(sampling_method=self.args.sampling_method, num_steps=self.args.inference_steps)
                sample_fn_reverse = self.transport_sampler.sample_ode_likelihood(sampling_method=self.args.sampling_method, num_steps=self.args.inference_steps, reverse=True)
            else:
                with torch.no_grad(): sample_fn = self.transport_sampler.sample_ode(sampling_method=self.args.sampling_method, num_steps=self.args.inference_steps)  # default to ode
        else:
            if self.args.likelihood:
                raise Exception("Not implemented")
            with torch.no_grad(): sample_fn = self.transport_sampler.sample_sde(num_steps=self.args.inference_steps, diffusion_form=self.args.diffusion_form, diffusion_norm=torch.tensor(self.args.diffusion_norm))

        if self.args.guided:
            with torch.no_grad(): samples = sample_fn(
                    zs,
                    partial(self.guided_velocity, **prep['model_kwargs'])
                )[-1]
        else:
            if self.args.likelihood:
                zs = zs.detach().requires_grad_(True)
                samples_logp, samples = sample_fn(
                    zs,
                    partial(self.model.forward_inference, **prep['model_kwargs'])
                )
            else:
                samples = sample_fn(
                    zs,
                    partial(self.model.forward_inference, **prep['model_kwargs'])
                )[-1]
        
        if self.args.design:
            # vector_out = samples[..., :-self.args.num_species]
            vector_out = prep["model_kwargs"]["x_now"]
            logits = samples[..., -self.args.num_species:]
        else:
            # print("WARNNING::")
            # print("Applying the following mask to the output vector:")
            # print(prep["model_kwargs"]['v_mask'])
            vector_out = samples *prep["model_kwargs"]['v_mask'] + prep["latents"]*(1-prep["model_kwargs"]['v_mask'])
            vector_out = vector_out.detach().requires_grad_(True)
            reverse_samples_logp, samples_zs = sample_fn_reverse(
                    vector_out,
                    partial(self.model.forward_inference, **prep['model_kwargs'])
                )

        if self.args.design:
            aa_out = torch.argmax(logits, -1)
            # aa_out = logits
        else:
            aa_out = torch.argmax(batch['species'], -1)
            # aa_out = batch['species']
        print('Time =', time.time()-s_time)
        if self.args.likelihood:
            return samples_logp, vector_out, aa_out, reverse_samples_logp, samples_zs
        else:
            return vector_out, aa_out
    
