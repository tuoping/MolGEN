from argparse import ArgumentParser
import os, json


def parse_train_args():
    parser = ArgumentParser()

    ## Trainer settings
    parser.add_argument("--ckpt", type=str, default=None)
    parser.add_argument("--validate", action='store_true', default=False)
    parser.add_argument("--num_workers", type=int, default=0)
    
    ## Epoch settings
    group = parser.add_argument_group("Epoch settings")
    group.add_argument("--epochs", type=int, default=1000)
    group.add_argument("--overfit", action='store_true')
    group.add_argument("--train_batches", type=int, default=None)
    group.add_argument("--val_batches", type=int, default=None)
    group.add_argument("--val_repeat", type=int, default=5)
    group.add_argument("--inference_batches", type=int, default=0)
    group.add_argument("--batch_size", type=int, default=1)
    group.add_argument("--val_freq", type=float, default=None)
    group.add_argument("--val_epoch_freq", type=int, default=1)
    group.add_argument("--no_validate", action='store_true')
    group.add_argument("--designability_freq", type=int, default=1)

    ## Logging args
    group = parser.add_argument_group("Logging settings")
    group.add_argument("--print_freq", type=int, default=100)
    group.add_argument("--ckpt_freq", type=int, default=10)
    group.add_argument("--wandb", action="store_true")
    group.add_argument("--run_name", type=str, default="default")
    

    ## Optimization settings
    group = parser.add_argument_group("Optimization settings")
    group.add_argument("--accumulate_grad", type=int, default=1)
    group.add_argument("--grad_clip", type=float, default=1.)
    group.add_argument("--check_grad", action='store_true')
    group.add_argument('--grad_checkpointing', action='store_true')
    group.add_argument('--adamW', action='store_true')
    group.add_argument('--ema', action='store_true')
    group.add_argument('--ema_decay', type=float, default=0.999)
    group.add_argument("--lr", type=float, default=1e-3)
    group.add_argument("--lr_decay", action='store_true')
    group.add_argument('--precision', type=str, default='32-true')
    
    ## Training data 
    group = parser.add_argument_group("Training data settings")
    # group.add_argument('--data_dir', type=str, default="tests/test_data/Transition1x")
    group.add_argument(
        '--data_dir',
        type=str,
        nargs='+',
        default=["tests/test_data/Transition1x"]
    )
    group.add_argument('--num_frames', type=int, default=1)
    group.add_argument('--suffix', type=str, default='')

    ### Masking settings
    group = parser.add_argument_group("Masking settings")
    group.add_argument('--localmask', action='store_true')

    ## Ablations settings
    group = parser.add_argument_group("Ablations settings")
    
    
    ## Model settings
    group = parser.add_argument_group("Model settings")
    group.add_argument('--dropout', type=float, default=0.0)

    group.add_argument('--time_multiplier', type=float, default=100.)
    group.add_argument('--abs_pos_emb', action='store_true')
    group.add_argument('--abs_time_emb', action='store_true')
    group.add_argument('--object_aware', action='store_true')

    group = parser.add_argument_group("Transport arguments")
    group.add_argument("--path-type", type=str, default="Linear", choices=["Linear", "GVP", "VP", "Schrodinger_Linear", "Schrodinger_Linear_onemodel"])
    group.add_argument("--last-step", type=str, default=None, choices=['Mean'])
    group.add_argument("--prediction", type=str, default="velocity", choices=["velocity", "score", "noise"])
    group.add_argument("--KL", type=str, default="L1", choices=['forward', 'reverse', 'symm', "L2", 'L1', 'alpha', 'score'])
    group.add_argument("--pref_symmkl", type=float, default=1)
    group.add_argument("--TSMloss", action='store_true')
    group.add_argument("--pref_TSMloss", type=float, default=1E-2)
    group.add_argument("--pref_loss_SDE", type=float, default=1E-2)
    group.add_argument('--uniform_prior', action='store_true')
    group.add_argument('--cubic_priorcell', action='store_true')
    group.add_argument("--std_cell_diagonal", type=float, default=2)
    group.add_argument("--std_cell_offdiagonal", type=float, default=0.2)
    group.add_argument('--target-std', type=float, default=0.)
    group.add_argument('--prior-std', type=float, default=1.)
    group.add_argument("--sampling_method", type=str, default="euler", choices=["dopri5", 'rk4', "euler", "Heun"])
    group.add_argument('--K_hutchinson_probe', type=float, default=4)
    group.add_argument('--K_hutchinson_probe_chunk', type=float, default=2)
    group.add_argument('--alpha_max', type=float, default=8)
    group.add_argument('--discrete_loss_weight', type=float, default=0.5)
    group.add_argument("--dirichlet_flow_temp", type=float, default=1.0)
    group.add_argument('--allow_nan_cfactor', action='store_true')
    group.add_argument('--x0std', type=float, default=None)
    group.add_argument('--loss_consistency', action='store_true')
    group.add_argument('--loss_graph', action='store_true')
    group.add_argument("--pref_loss_graph", type=float, default=1)
    group.add_argument(
        "--loss_line_graph",
        action="store_true",
        help=(
            "Also supervise bond angles formed by pairs of reference-graph "
            "edges that share a central atom."
        ),
    )
    group.add_argument(
        "--pref_loss_line_graph",
        type=float,
        default=1.0,
        help="Weight of the bond-angle line-graph loss.",
    )
    group.add_argument(
        "--graph_loss_mode",
        type=str,
        default="legacy",
        choices=["legacy", "euler"],
        help=(
            "legacy reproduces the original x0 + t*v graph objective; "
            "euler compares deterministic one-step coordinate changes."
        ),
    )
    group.add_argument(
        "--graph_time_power",
        type=float,
        default=1.0,
        help=(
            "Euler graph-loss weight t**graph_time_power. Larger values "
            "suppress early-time graph errors more strongly."
        ),
    )

    group.add_argument('--loss_collision', action='store_true')
    group.add_argument(
        "--pref_loss_collision",
        type=float,
        default=1.0,
        help="Weight of the late-time collision loss.",
    )
    group.add_argument(
        "--collision_min_distance",
        type=float,
        default=0.8,
        help="Fallback collision cutoff when atomic species are unavailable.",
    )
    group.add_argument(
        "--collision_radius_scale",
        type=float,
        default=0.8,
        help="Pair cutoff as a multiple of the summed covalent radii.",
    )
    group.add_argument("--collision_softness", type=float, default=0.05)
    group.add_argument(
        "--collision_start_time",
        type=float,
        default=0.75,
        help=(
            "Time at which the collision loss starts to turn on. It is zero "
            "at and below this time."
        ),
    )
    group.add_argument(
        "--collision_full_time",
        type=float,
        default=0.95,
        help=(
            "Time at which the collision loss reaches full weight. A smooth "
            "ramp is used after collision_start_time."
        ),
    )
    group.add_argument('--beta_sample_t', type=float, default=0.8)
    group.add_argument("--loss-weight", type=str, default=None, choices=["None", "velocity", "likelihood"])
    group.add_argument('--weight_loss_var_x0', type=float, default=0)
    group.add_argument("--likelihood", type=str, default=None, choices = ["None", "EJE", "FND"])

    # ## Diffusion settings
    group = parser.add_argument_group("Diffusion settings")
    group.add_argument("--diffusion-form", type=str, default="constant", choices=["constant" ,'increasing-decreasing', 'decreasing'])
    group.add_argument("--diffusion-norm", type=float, default=1E-4)
    

    ## video settings
    group = parser.add_argument_group("Video settings")
    group.add_argument('--tps_condition', action='store_true')
    group.add_argument('--design', action='store_true')
    group.add_argument('--design_from_traj', action='store_true')
    group.add_argument('--sim_condition', action='store_true')
    group.add_argument('--frame_interval', type=int, default=1)
    group.add_argument('--ratio_conditional', type=float, default=1.0)
    group.add_argument('--cond_interval', type=int, default=None) # for superresolution
    
    ## Equivariant Transformer settings
    group.add_argument('--num_species', type=int, default=80)
    group.add_argument('--embed_dim', type=int, default=128)
    group.add_argument('--edge_dim', type=int, default=128)
    group.add_argument('--num_convs', type=int, default=5)
    group.add_argument('--num_heads', type=int, default=8)
    group.add_argument('--ff_dim', type=int, default=128)
    group.add_argument('--cutoff', type=float, default=5)
    
    ## nonequil. simulation settings
    group.add_argument('--potential_model', action='store_true')
    # group.add_argument("--pbc", action='store_true')
    group.add_argument("--guided", action='store_true')
    group.add_argument("--latt_path", action="store_true")
    group.add_argument("--guidance", action="store_true")

    group.add_argument("--inference_steps", type=int, default=20)

    args = parser.parse_args()
    if not (
        0.0 <= args.collision_start_time
        < args.collision_full_time
        <= 1.0
    ):
        parser.error(
            "collision times must satisfy 0 <= collision_start_time "
            "< collision_full_time <= 1"
        )

    if args.val_repeat < 1:
        parser.error("val_repeat must be a positive integer")
    if args.loss_line_graph and not args.loss_graph:
        parser.error("--loss_line_graph requires --loss_graph")
    os.environ["MODEL_DIR"] = os.path.join("workdir", args.run_name)
    if args.loss_weight == "None":
        args.loss_weight = None
    if args.likelihood == "None":
        args.likelihood = None
    args.pbc = True
    
    os.makedirs(f"workdir/{args.run_name}", exist_ok=True)
    with open(f"workdir/{args.run_name}/args.json", "w") as f:
        json.dump(vars(args), f, indent=4)
    return args
