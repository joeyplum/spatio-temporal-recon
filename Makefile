.PHONY: mamba pip clean

mamba:
	mamba env create -f environment.yaml

update:
	mamba env update --file environment.yaml --prune

pip:
	pip install git+https://github.com/mikgroup/sigpy.git@main
	pip install opencv-python
	pip install nibabel
	pip install tk
	pip install antspyx

clean:
	rm -rf __pycache__
	rm -rf .ipynb_checkpoints
	mamba env remove -n spatio-temporal-recon
