# =============================================================================
# main.py
# Single entry point for the entire project.
# Routes to train or detect mode based on the --mode argument.
#
# Usage examples:
#   python main.py --mode train --model custom
#   python main.py --mode train --model mobilenet
#   python main.py --mode train --model vgg16
#   python main.py --mode detect --model mymodel_custom.h5
#   python main.py --mode detect --model mymodel_mobilenet.h5
# =============================================================================

import argparse


def main():
    parser = argparse.ArgumentParser(
        description='Face Mask Detector — train a model or run live detection.',
        formatter_class=argparse.RawTextHelpFormatter
    )

    parser.add_argument(
        '--mode',
        choices=['train', 'detect'],
        required=True,
        help=(
            'train  → train a model and save it as .h5\n'
            'detect → run live webcam detection using a saved model'
        )
    )

    parser.add_argument(
        '--model',
        type=str,
        default='custom',
        help=(
            'For --mode train:  architecture to use (custom | mobilenet | vgg16)\n'
            'For --mode detect: path to the .h5 model file\n'
            'Default: custom'
        )
    )

    args = parser.parse_args()

    if args.mode == 'train':
        # Map model name to builder function and save path
        model_map = {
            'custom':    ('model',           'build_custom_cnn',  'mymodel_custom.h5'),
            'mobilenet': ('model_mobilenet', 'build_mobilenet',   'mymodel_mobilenet.h5'),
            'vgg16':     ('model_vgg16',     'build_vgg16',       'mymodel_vgg16.h5'),
        }

        if args.model not in model_map:
            print(f"Unknown model '{args.model}'. Choose from: custom, mobilenet, vgg16")
            return

        module_name, func_name, save_path = model_map[args.model]

        # Dynamically import the right module and function
        import importlib
        module = importlib.import_module(module_name)
        build_fn = getattr(module, func_name)

        model = build_fn()
        model.summary()

        from train import run_training
        run_training(model, save_path)

    elif args.mode == 'detect':
        # For detect mode, --model should be the path to the .h5 file
        model_path = args.model

        # If user passed architecture name instead of file path, map it
        default_paths = {
            'custom':    'mymodel_custom.h5',
            'mobilenet': 'mymodel_mobilenet.h5',
            'vgg16':     'mymodel_vgg16.h5',
        }
        if model_path in default_paths:
            model_path = default_paths[model_path]

        from detector import run_detection
        run_detection(model_path)


if __name__ == '__main__':
    main()
