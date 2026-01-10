# Run Commands

python3 softdtw_nn_pipeline.py build-index --dataset ./sgsl_dataset --out ./processed/prototypes --n-aug 200

python softdtw_nn_pipeline.py classify --protos ./processed/prototypes --query ./sgsl_dataset/abuse/pose.pkl --k 5

Folders with no .pkl files:
find /Users/chaitanya/Documents/Coding/singapore-sign-language/sgsl_dataset -type d -mindepth 1 -maxdepth 1 ! -exec test -f {}/pose.pkl \; -print

