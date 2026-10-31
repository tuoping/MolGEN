from .ema import ExponentialMovingAverage
from .logger import get_logger

logger = get_logger(__name__)

import torch, time
from torch import nn
import copy
import numpy as np
import math
from functools import partial

from .model.equivariant_latent_model import EquivariantTransformer_dpm, Encoder_dpm, Processor, Decoder
from .wrapper import Wrapper, gather_log, get_log_mean
from .transport.utils import collision_loss


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



@torch.no_grad()
def lattice_polar_build_torch(k):
    assert k.dim() == 2, "input must be batched k of shape (B,6)"
    S0 = torch.stack([k[:, 3] + k[:, 4] + k[:, 5], k[:, 0], k[:, 1]], dim=1)  # (B, 3)
    S1 = torch.stack([k[:, 0], -k[:, 3] + k[:, 4] + k[:, 5], k[:, 2]], dim=1)  # (B, 3)
    S2 = torch.stack([k[:, 1], k[:, 2], -2 * k[:, 4] + k[:, 5]], dim=1)  # (B, 3)
    S = torch.stack([S0, S1, S2], dim=1)  # (B, 3, 3)
    expS = torch.matrix_exp(S)  # (B, 3, 3)
    return expS


def decompose_symmetric_matrix(S: torch.Tensor):
    k0 = S[:, 0, 1]
    k1 = S[:, 0, 2]
    k2 = S[:, 1, 2]
    k3 = (S[:, 0, 0] - S[:, 1, 1]) / 2
    k4 = (S[:, 0, 0] + S[:, 1, 1] - 2 * S[:, 2, 2]) / 6
    k5 = (S[:, 0, 0] + S[:, 1, 1] + S[:, 2, 2]) / 3
    k = torch.vstack([k0, k1, k2, k3, k4, k5]).transpose(-1, -2)
    return k

@torch.no_grad()
def lattice_polar_decompose_torch(lattices: torch.Tensor):
    assert lattices.dim() == 3, "input must be batched lattices of shape (B,3,3)"
    A, U = torch.linalg.eigh(lattices @ lattices.transpose(-1,-2))  # J = L^T @ L
    # S = 1/2 U log(A) U^T
    A = torch.diag_embed(A.log()) / 2
    S = U @ A @ U.transpose(-1, -2)
    k = decompose_symmetric_matrix(S)
    return k

from .spring import NNSpring
from .spring import ZBLRepulsiveWall
from .transport.utils import collision_loss, graph_loss
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
        num_vector_out=1
        if args.design:
            num_scalar_out = self.args.num_species
        else:
            num_scalar_out = 0
            
        latent_dim = args.embed_dim
        
        if args.tps_condition:
            encoder = Encoder_dpm(num_species, latent_dim, (64+48+8)*3, latent_dim, input_dim=1, cv_dim=1, object_aware=args.object_aware)
        elif args.sim_condition:
            encoder = Encoder_dpm(num_species, latent_dim, (64+48+8)*2, latent_dim, input_dim=1, cv_dim=1, object_aware=args.object_aware)
        else:
            encoder = Encoder_dpm(num_species, latent_dim, (64+48+8), latent_dim, input_dim=1, cv_dim=1, object_aware=args.object_aware)

        processor = Processor(num_convs=5, node_dim=latent_dim, num_heads=8, ff_dim=args.ff_dim, edge_dim=latent_dim)
        print("Initializing drift model")
        latt_path = args.latt_path

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
        else:
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
                latt_path = latt_path
            )
            if args.path_type == "Schrodinger_Linear":
                print("Initializing score model")
                self.score_model = EquivariantTransformer_dpm(
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
                    latt_path = latt_path
                )
            else:
                self.score_model = None
            from .model.polynomialRepulsiveEnergy import PolynomialRepulsiveEnergy
            self.transport = create_transport(
                args,
                args.path_type,
                args.prediction,
                train_eps=1e-5,
                sample_eps=1e-5,
                score_model=self.score_model,
                latt_path = latt_path,
                weightfunction_x=PolynomialRepulsiveEnergy(1.4, prefactor=10, n_pow=4)
            )
            if self.transport.latt_path:
                self.transport.mean_atomic_volume = getattr(args, "mean_atomic_volume", None)
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

    def load_state_dict(self, state_dict, strict=True):
        return super().load_state_dict(state_dict, strict=False)

    def on_validation_epoch_end(self):
        if self.args.ema:
            self.restore_cached_weights()
        log = self._log
        log = {key: log[key] for key in log if "val_" in key}
        log = gather_log(log, self.trainer.world_size)
        mean_log = get_log_mean(log)
        self.log("val_loss", mean_log['val_loss'])
        if self.args.KL == 'symm':
            self.log("val_loss_symmkl", mean_log['val_loss_symmkl'])
        self.log("val_rollout_graph_err", mean_log['val_rollout_graph_err'])
        if self.args.loss_collision:
            self.log("val_loss_collision", mean_log['val_loss_collision'])
            self.log("val_collision_count", mean_log['val_collision_count'])
            self.log("val_collision_rate", mean_log['val_collision_rate'])
            self.log(
                "val_collision_min_distance",
                mean_log['val_collision_min_distance'],
            )
        self.log(
            "val_rollout_collision_count",
            mean_log['val_rollout_collision_count'],
        )
        self.log(
            "val_rollout_collisions_per_structure",
            mean_log['val_rollout_collisions_per_structure'],
        )
        self.log(
            "val_rollout_collision_min_distance",
            mean_log['val_rollout_collision_min_distance'],
        )
        if self.transport.latt_path:
            self.log('val_loss_lattflow', mean_log['val_loss_lattflow'])
        if not self.args.potential_model:
            if not self.args.latt_path:
                self.log("val_meanRMSD", mean_log['val_meanRMSD'])
            self.log("val_E_spring", mean_log['val_E_spring'])
        if self.args.path_type in ["Schrodinger_Linear", "Schrodinger_Linear_onemodel"]:
            self.log("val_loss_path", mean_log['val_loss_path'])
        self.print_log(prefix='val', save=False)

    def prep_batch(self, batch):
        if self.args.potential_model:
            return self.prep_batch_x_potential_model(batch)
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

        B, T, L, num_elem = species.shape

        v_loss_mask = batch["v_mask"]

        ### for schrodinger bridge
        if not self.args.potential_model:
            self.transport.x0std = batch['x0std']

        _cell = batch['cell']
        if self.args.latt_path and self.args.cubic_priorcell:
            _volume = torch.linalg.det(_cell).abs().unsqueeze(-1).unsqueeze(-1)
            self.transport.prior_cell = (torch.eye(3).unsqueeze(0).unsqueeze(0).to(_cell.device) * _volume**(1./3)).to(_TORCH_FLOAT_PRECISION)
        else:
            self.transport.prior_cell = batch["cell"].to(_TORCH_FLOAT_PRECISION)

        if not self.args.uniform_prior:
            self.transport.prior_mean = batch['x']

        B, T, L, _ = latents.shape
        assert _ == 3, f"latents shape should be (B, T, D, 3), but got {latents.shape}"
        ########
        
        if "inpainting_mask" not in batch.keys():
            batch['inpainting_mask'] = torch.ones(B,T,L, dtype=int, device=species.device)
            batch['inpainting_v_mask'] = torch.ones(B,T,L,3, dtype=int, device=species.device)

        atomic_numbers = batch['atomic_numbers']
        nn_spring = NNSpring(latents.view(B*T,L,3), batch['cell'].view(B*T,3,3), atomic_numbers=atomic_numbers.view(B*T, L))
        if "x1std" in batch:
            inv_cell = torch.linalg.inv(_cell)
            noise = torch.randn_like(latents) * batch['x1std'][...,None,None]
            latents = latents + noise @ inv_cell

        conditional_batch = None
        data = {
                    "species": species.to(_TORCH_FLOAT_PRECISION),
                    "latents": latents.to(_TORCH_FLOAT_PRECISION),
                    'loss_mask': v_loss_mask.to(_TORCH_FLOAT_PRECISION),
                    'model_kwargs': {
                        "cv": None,
                        "aatype": species.to(_TORCH_FLOAT_PRECISION),
                        'x1': latents.to(_TORCH_FLOAT_PRECISION),
                        'v_mask': (v_loss_mask!=0).to(int),
                        "cell": batch['cell'].to(_TORCH_FLOAT_PRECISION),
                        "num_atoms": batch["num_atoms"],
                        "dt": torch.zeros(B,T,1, dtype=_TORCH_FLOAT_PRECISION, device=species.device),
                        "conditions": None
                    },
                    'conditional_batch': conditional_batch
                }
        

        wall = ZBLRepulsiveWall()
        
        with torch.enable_grad():
            if "forces" in batch:
                data["forces"] = batch['forces'].to(_TORCH_FLOAT_PRECISION) + wall.build_force(latents.view(B*T,L,3), batch['cell'].view(B*T,3,3), atomic_numbers.view(B*T,L)).to(_TORCH_FLOAT_PRECISION).view(B,T,L,3)
            else:
                data["forces"] = -noise / batch['x1std'][...,None,None]**2 + wall.build_force(latents.view(B*T,L,3), batch['cell'].view(B*T,3,3), atomic_numbers.view(B*T,L)).to(_TORCH_FLOAT_PRECISION).view(B,T,L,3)
        if not self.args.potential_model:
            data['x0std'] = batch['x0std'].to(_TORCH_FLOAT_PRECISION)
        return data

    def prep_batch_x_potential_model(self, batch):
        species = batch["species"]
        latents = batch["x"]
        B, T, L, num_elem = species.shape

        v_loss_mask = batch["v_mask"]

        B, T, L, _ = latents.shape
        assert _ == 3, f"latents shape should be (B, T, D, 3), but got {latents.shape}"
        ########
        
        if "inpainting_mask" not in batch.keys():
            batch['inpainting_mask'] = torch.ones(B,T,L, dtype=int, device=species.device)
            batch['inpainting_v_mask'] = torch.ones(B,T,L,3, dtype=int, device=species.device)

        conditional_batch = None
        data = {
                    "species": species.to(_TORCH_FLOAT_PRECISION),
                    "latents": latents.to(_TORCH_FLOAT_PRECISION),
                    'loss_mask': v_loss_mask.to(_TORCH_FLOAT_PRECISION),
                    'model_kwargs': {
                        "cv": None,
                        "aatype": species.to(_TORCH_FLOAT_PRECISION),
                        'x1': latents.to(_TORCH_FLOAT_PRECISION),
                        'v_mask': (v_loss_mask!=0).to(int),
                        "cell": batch['cell'].to(_TORCH_FLOAT_PRECISION),
                        "num_atoms": batch["num_atoms"],
                        "dt": torch.zeros(B,T,1, dtype=_TORCH_FLOAT_PRECISION, device=species.device),
                        "conditions": None
                    },
                    'conditional_batch': conditional_batch
                }
        
        if "forces" in batch:
            data["forces"] = batch['forces'].to(_TORCH_FLOAT_PRECISION)
        if "E" in batch:
            data["E"] = batch['E'].to(_TORCH_FLOAT_PRECISION)
        return data
    
    def general_step(self, batch, stage='train'):
        self.iter_step += 1
        self.stage = stage
        start1 = time.time()
        prep = self.prep_batch(batch)
        if stage == "val" and self.args.latt_path:
            def clone_batch(x):
                if torch.is_tensor(x):
                    return x.clone()
                elif isinstance(x, dict):
                    return {k: clone_batch(v) for k, v in x.items()}
                elif isinstance(x, list):
                    return [clone_batch(v) for v in x]
                elif isinstance(x, tuple):
                    return tuple(clone_batch(v) for v in x)
                else:
                    return x
            batch_val = clone_batch(batch)
            prep_val = self.prep_batch(batch_val)
        else:
            batch_val = batch
            prep_val = prep
        start = time.time()

        if self.args.potential_model:
            B,T,L,_ = prep["latents"].shape
            t = torch.ones((B,), device=prep["latents"].device).to(_TORCH_FLOAT_PRECISION)
            with torch.enable_grad():
                xfrac = prep['latents'].detach().requires_grad_(True)
                energy = self.potential_model(xfrac, t, **prep["model_kwargs"])
                energy = energy.sum(dim=2).squeeze(-1)
                
                grad_frac = torch.autograd.grad(
                    energy.sum(),
                    xfrac,
                    create_graph=(stage == "train"),
                    retain_graph=(stage == "train"),
                )[0]
                grad_cart = torch.einsum(
                    "btni,btij->btnj",
                    grad_frac,
                    torch.linalg.inv(prep["model_kwargs"]['cell']).transpose(-1, -2),
                )
            loss_energy = ((energy -prep["E"]).abs()).sum(-1)
            self.prefix_log('loss_energy', loss_energy.detach().cpu())      
            forces = -grad_cart
            loss_forces = (forces - prep['forces']).norm(dim=-1).sum(dim=-1)
            self.prefix_log('loss_forces', loss_forces.detach().cpu())  
            loss = loss_forces + loss_energy
        else:
            forces = None
            if self.args.path_type in ["Schrodinger_Linear", "Schrodinger_Linear_onemodel"]:
                forces = prep['forces']
            x0std = self.args.x0std
            if "x0std" in prep:
                x0std = prep['x0std']

            out_dict = self.transport.training_losses(
                model=self.model,
                x1=prep['latents'],
                aatype1=batch['species'],
                atomic_numbers=batch.get('atomic_numbers'),
                mask=prep['loss_mask'],
                model_kwargs=prep['model_kwargs'],
                forces = forces,
                x0std=x0std,
                global_step = self.current_epoch
            )
            self.prefix_log('model_dur', time.time() - start)
            self.prefix_log('time', out_dict['t'].detach().cpu())
            # self.prefix_log('conditional_batch', prep['conditional_batch'].to(torch.float32))
            loss_gen = out_dict['loss']
            self.prefix_log('loss_gen', loss_gen.detach().cpu())
            assert self.args.weight_loss_var_x0 == 0
            loss = loss_gen
            if self.args.path_type in ["Schrodinger_Linear", "Schrodinger_Linear_onemodel"]:
                self.prefix_log("loss_dsm", out_dict['loss_dsm'].detach().cpu())
                if self.args.TSMloss:
                    self.prefix_log("loss_tsm_0", out_dict['loss_tsm_0'].detach().cpu())
                    self.prefix_log("loss_tsm_1", out_dict['loss_tsm_1'].detach().cpu())
                self.prefix_log("loss_path", out_dict['loss_dsm'].detach().cpu()+out_dict['loss_flow'].detach().cpu())
            if self.args.KL == 'symm':
                self.prefix_log('loss_symmkl', out_dict['loss_symmkl'].detach().cpu())
                self.prefix_log('loss_l1', out_dict['loss_l1'].detach().cpu())
            if self.args.KL == 'alpha':
                self.prefix_log('loss_alphadiv', out_dict['loss_alphadiv'].detach().cpu())
                self.prefix_log('loss_l1', out_dict['loss_l1'].detach().cpu())
            if self.args.KL == 'score':
                self.prefix_log('loss_score', out_dict['loss_score'].detach().cpu())
                self.prefix_log('loss_l1', out_dict['loss_l1'].detach().cpu())
            if self.args.loss_consistency:
                self.prefix_log('loss_consistency', out_dict['loss_consistency'].detach().cpu())
            if self.args.loss_graph:
                self.prefix_log('loss_graph', out_dict['loss_graph'].detach().cpu())
                if getattr(self.args, "loss_line_graph", False):
                    self.prefix_log(
                        'loss_line_graph',
                        out_dict['loss_line_graph'].detach().cpu(),
                    )
            if self.args.loss_collision:
                self.prefix_log(
                    'loss_collision',
                    out_dict['loss_collision'].detach().cpu(),
                )
                self.prefix_log(
                    'collision_count',
                    out_dict['collision_count'].detach().cpu(),
                )
                self.prefix_log(
                    'collision_rate',
                    out_dict['collision_rate'].detach().cpu(),
                )
                self.prefix_log(
                    'collision_min_distance',
                    out_dict['collision_min_distance'].detach().cpu(),
                )

            if self.args.design:
                self.prefix_log('loss_logit', out_dict['loss_logit'].detach().cpu())

            self.prefix_log('model_dur', time.time() - start)
            self.prefix_log("loss_flow", out_dict['loss_flow'].detach().cpu())
            if self.transport.latt_path:
                self.prefix_log('loss_lattflow', out_dict['loss_lattflow'].detach().cpu())
            if stage == "val":
                self._val_rmse(batch_val, prep_val)
                
        self.prefix_log('loss', loss.detach().cpu())
        self.prefix_log('dur', time.time() - self.last_log_time)
        if 'name' in batch:
            self.prefix_log('name', ','.join(batch['name']))
        self.prefix_log('general_step_dur', time.time() - start1)
        self.last_log_time = time.time()

        if not torch.isfinite(loss.mean()):
            return None
        if torch.isnan(loss.mean()):
            return None
        return loss.mean()


    def _val_rmse(self, batch, prep, stage="val"):
        species = [1, 3, 5, 6, 7, 8, 9, 11, 12, 13, 14, 15, 16, 17, 19, 20, 21,
                                            22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 37, 39, 40,
                                            41, 42, 44, 45, 46, 47, 48, 49, 51, 53, 55, 57, 58, 59, 60, 62, 63,
                                            64, 65, 66, 67, 68, 69, 70, 71, 72, 74, 77, 78, 79, 80, 82, 83, 90,
                                            92, 93, 94]
        B,T,L,_ = prep['latents'].shape
        try:
            if self.args.latt_path:
                pred_pos, _, pred_cell = self.inference(batch, stage=stage)
                cell = pred_cell[-1]
            else:
                pred_pos, _ = self.inference(batch, stage=stage)
                cell = prep['model_kwargs']['cell']
            ref_pos = prep['latents']
            ref_cell = batch['cell']
            with torch.no_grad():
                graph_err, graph_info = graph_loss(
                    pred_pos[-1].view(B * T, L, 3),
                    ref_pos.view(B * T, L, 3),
                    cell.view(B * T, 3, 3),
                    self.args.cutoff,
                    reference_cell=ref_cell,
                    use_reference_image=False,
                    compute_line_graph=False,
                )
                self.prefix_log('rollout_graph_err', graph_err)
                if not self.args.latt_path:
                    ## (\Delta d per atom) # B,T,L
                    # x, x_ref: (..., N, 3), fractional coordinates
                    # RMSD = sqrt(mean_i |dx_i|^2)
                    dx = pred_pos[-1] - ref_pos
                    dx = dx - torch.round(dx)       # minimum image in fractional coords
                    dx_cart = dx @ cell             # cell: (..., 3, 3) or (3, 3)
                    err = torch.sqrt((dx_cart**2).sum(dim=-1).mean(dim=-1))
                    ## mean RMSD per sample # B
                    err = err.mean(dim=-1)
                    self.prefix_log('meanRMSD', err)  # An extra factor of 3 was divided when taking the mean over the T dimension
                if 'atomic_numbers' in batch:
                    atomic_numbers = batch['atomic_numbers'].to(
                        device=pred_pos.device,
                        dtype=torch.long,
                    ).reshape(B * T, L)
                else:
                    labels_B = torch.argmax(batch['species'], dim=-1)
                    atomic_numbers = torch.tensor(
                        [
                            [species[label] for label in labels.flatten()]
                            for labels in labels_B
                        ],
                        device=pred_pos.device,
                        dtype=torch.long,
                    ).reshape(B * T, L)
                generated_frac = pred_pos[-1].reshape(B * T, L, 3)
                reference_frac = ref_pos.reshape(B * T, L, 3)
                generated_cell = cell.reshape(B * T, 3, 3)
                reference_cell = ref_cell.reshape(
                    B * T, 3, 3
                )
                # nn_spring = NNSpring(
                #     reference_frac,
                #     generated_cell,
                #     atomic_numbers=atomic_numbers,
                #     reference_cell=reference_cell,
                # )
                wall = ZBLRepulsiveWall()
                energies_B = (
                    wall.build_energy(
                        generated_frac,
                        generated_cell,
                        atomic_numbers,
                    )
                )
                # energies_B[i] = e.sum()
                self.prefix_log("E_spring", energies_B)
                _, collision_info = collision_loss(
                    pred_pos[-1],
                    cell,
                    atomic_numbers=atomic_numbers,
                    min_distance=getattr(
                        self.args, "collision_min_distance", 0.8
                    ),
                    radius_scale=getattr(
                        self.args, "collision_radius_scale", 0.8
                    ),
                    softness=getattr(
                        self.args, "collision_softness", 0.05
                    ),
                    atom_mask=batch['species'].sum(dim=-1) > 0,
                )
                rollout_collision_count = torch.as_tensor(
                    collision_info['num_collisions'],
                    dtype=pred_pos.dtype,
                    device=pred_pos.device,
                )
                self.prefix_log(
                    'rollout_collision_count', rollout_collision_count
                )
                self.prefix_log(
                    'rollout_collisions_per_structure',
                    rollout_collision_count / max(B * T, 1),
                )
                self.prefix_log(
                    'rollout_collision_min_distance',
                    collision_info['min_distance'],
                )
        except:
            print("WARNNING:: Inference failed !!!")
            self.prefix_log('meanRMSD', torch.nan)
            self.prefix_log("E_spring", torch.nan)
            if self.args.loss_graph:
                self.prefix_log('rollout_graph_err', torch.nan)
                self.prefix_log('rollout_collision_count', torch.nan)
                self.prefix_log(
                    'rollout_collisions_per_structure', torch.nan
                )
                self.prefix_log(
                    'rollout_collision_min_distance', torch.nan
                )


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

    
    def inference(self, batch, stage='inference', sample_latt=True):
        s_time= time.time()
        self.stage = stage
        prep = self.prep_batch(batch)
        # if self.args.uniform_prior and not self.args.cubic_priorcell:
        #     self.transport.prior_mean = batch['x']
        latents = prep['latents']
        B, T, N, D = latents.shape
        x0std = self.args.x0std
        if "x0std" in prep:
            x0std = prep['x0std']

        if self.args.design:
            zs_discrete = torch.distributions.Dirichlet(torch.ones(B*T, N, self.args.num_species, device=latents.device)).sample()
            zs_discrete = zs_discrete.view(B, T, N, self.args.num_species)


        _, zs, zs_mean = self.transport.sample(latents.shape, self.device, x0std)
        zs = zs[0]
        zs_mean = zs_mean[0]
        if self.transport.latt_path:
            if sample_latt:
                cell0 = self.transport.sample_latt(zs.shape, self.device)
            else:
                cell0 = self.transport.prior_cell.clone()
        self.integration_step = 0
        if self.args.path_type not in ["Schrodinger_Linear", "Schrodinger_Linear_onemodel"]:
            # if self.args.likelihood == "EJE":
            match self.args.likelihood:
                case "EJE":
                    raise Exception("Not running likelihood")
                case None:
                    with torch.no_grad(): sample_fn = self.transport_sampler.sample_ode(sampling_method=self.args.sampling_method, num_steps=self.args.inference_steps)  # default to ode
                case _:
                    raise Exception(f"Wrong likelihood parameter: {self.args.likelihood}")
        else:
            last_step = getattr(self.args, "last_step", None)
            match self.args.likelihood:
                case "FND":
                    raise Exception("Not running likelihood")
                case None:
                    score_model = self.score_model if self.score_model is not None else self.model
                    with torch.no_grad(): sample_fn = self.transport_sampler.sample_sde(num_steps=self.args.inference_steps, diffusion_form=self.args.diffusion_form, diffusion_norm=torch.tensor(self.args.diffusion_norm), score_model=partial(score_model.forward_inference, **prep['model_kwargs']), last_step=last_step, last_step_size=0.04 if last_step is not None else 0.0)
                case _:
                    raise Exception("Wrong likelihood argument (not implemented for SDE): "+self.args.likelihood)

        assert not self.args.guided


        match self.args.likelihood:
            case "EJE":
                K_hutchinson_probe = self.args.K_hutchinson_probe
                K_hutchinson_probe_chunk = self.args.K_hutchinson_probe_chunk
                assert K_hutchinson_probe % K_hutchinson_probe_chunk == 0
                def extend_kwargs(kargs):
                    _kargs = {}
                    for k in kargs.keys():
                        _v = kargs[k]
                        if isinstance(_v, dict):
                            _kargs[k] = {}
                            for _k in _v.keys():
                                _kargs[k][_k] = {}
                                for _k_2 in _v[_k].keys():
                                    _kargs[k][_k][_k_2] = (
                                        _v[_k][_k_2].detach()
                                         .unsqueeze(0)
                                         .repeat((K_hutchinson_probe_chunk,) + (1,) * _v[_k][_k_2].dim())
                                         .reshape(K_hutchinson_probe_chunk * B, *_v[_k][_k_2].shape[1:])
                                         .requires_grad_(False)
                                    )
                        else:
                            _kargs[k] = (
                                _v.detach()
                                 .unsqueeze(0)
                                 .repeat((K_hutchinson_probe_chunk,) + (1,) * _v.dim())
                                 .reshape(K_hutchinson_probe_chunk * B, *_v.shape[1:])
                                 .requires_grad_(False)
                            )
                    return _kargs

        if self.transport.latt_path and not self.args.design:
            zs = zs.detach()
            cell0 = cell0.detach()
            
            match self.args.likelihood:
                case "EJE":
                    samples_logp, samples, samples_logp_var = sample_fn(
                        (zs, cell0),
                        partial(self.model.forward_inference, **prep['model_kwargs'])
                    )                
                case _:
                    with torch.no_grad(): 
                        samples = sample_fn(
                            (zs, cell0),
                            partial(self.model.forward_inference, **prep['model_kwargs']),
                            **prep["model_kwargs"]
                        )
        elif self.args.design:
            zs = zs.detach()
            zs_discrete = zs_discrete.detach()
            with torch.no_grad(): 
                samples, a_samples = sample_fn(
                    (zs_discrete, zs),
                    partial(self.model.forward_inference, **prep['model_kwargs']),
                    **prep["model_kwargs"]
                )
        else:
            zs = zs.detach()
            match self.args.likelihood:
                case "EJE":
                    _model_kwargs = extend_kwargs(prep['model_kwargs'])
                    samples_logp, samples, samples_logp_var = sample_fn(
                        zs,
                        partial(self.model.forward_inference, **_model_kwargs)
                    )
                case "FND":
                    with torch.no_grad(): 
                        samples_logp, _samples_logp, samples = sample_fn(
                            zs,
                            partial(self.model.forward_inference, **prep['model_kwargs'])
                        )
                    _samples_logp = _samples_logp.detach().cpu()
                    samples_logp = samples_logp.detach().cpu()
                case None:
                    if self.args.guidance:
                        with torch.no_grad(): 
                            samples = sample_fn(
                                zs,
                                partial(self.model.forward_inference, **prep['model_kwargs']),
                                **prep["model_kwargs"]
                            )
                    else:
                        with torch.no_grad():
                            samples = sample_fn(
                                zs,
                                partial(self.model.forward_inference, **prep['model_kwargs']),
                            )
                case _:
                    raise Exception(f"Wrong likelihood parameter: {self.args.likelihood}")


        # print("WARNNING::")
        # print("Applying the following mask to the output vector:")
        if self.transport.latt_path:
            samples[0] = torch.stack(samples[0])
            samples[1] = torch.stack(samples[1])
            samples[0] = samples[0] *prep["model_kwargs"]['v_mask'] + prep["latents"]*(1-prep["model_kwargs"]['v_mask'])
        else:
            if isinstance(samples, list):
                samples = torch.stack(samples)
                samples = samples *prep["model_kwargs"]['v_mask'] + prep["latents"]*(1-prep["model_kwargs"]['v_mask'])

        print("WARNNING::")
        print("Not Applying Mask to output atom type samples:")
        if self.args.design:
            return samples, torch.stack(a_samples)
        else:
            aa_out = torch.argmax(batch['species'], -1)
            # aa_out = batch['species']
        print('Time =', time.time()-s_time)


        # if self.args.likelihood == "EJE":
        match self.args.likelihood:
            case "EJE":
                return (samples_logp, samples_logp_var), samples, aa_out, zs-zs_mean
            case "FND":
                if self.transport.latt_path:
                    raise Exception("FND for latt_path not implemented")
                return torch.concatenate([samples_logp, _samples_logp], dim=-1), samples, aa_out, zs-zs_mean
            case _:
                if self.transport.latt_path:
                    return samples[0], aa_out, samples[1]
                else:
                    return samples, aa_out
    
