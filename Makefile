.PHONY: conda pip clean

conda:
	conda env create -f environment.yaml

update:
	conda env update --file environment.yaml --prune

pip:
	pip install git+https://github.com/mikgroup/sigpy.git@main
	pip install opencv-python
	pip install nibabel
	pip install tk
	pip install antspyx
	git clone git@github.com:MattWill660/ReadPhilips.git

clean:
	rm -rf __pycache__
	rm -rf .ipynb_checkpoints
	conda env remove -n spatio-temporal-recon
