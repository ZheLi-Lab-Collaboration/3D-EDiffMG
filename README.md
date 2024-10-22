# 3D-EDiffMG：3D Equivariant Diffusion-Driven for Molecule Generation 

![overview_png](./img/pic1.png)

## Requirements

- rdkit>=2022.9.5
- openbabel>=3.0.0
- torch>=1.10.1
- torch-geometric>=2.1.0

## Getting start

### Data Preparation

* Clone the repsitory into the local folder:

  ```shell
  git clone git@github.com:ZheLi-Lab-Collaboration/3D-EDiffMG.git
  ```

* If you intend to use a dataset you have screened and ensure its file format is .XYZ

  ```shell
  python ./data_prepare/data/prepare/prepare_dataset.py
  ```

* MQ9 Dataset. Put them into `./data/QM9/qm9/raw`

  [QM9 Dataset](https://drive.google.com/file/d/1rgM77-2WqUzcQJENdfemooDNc2lOyRtf/view?usp=drive_link)

###  Training

- One GPU

  ```shell
  python EDiffMG_train.py --config './configs/qm9.yml'
  ```

- Multi-GPUs

  ```shell
  python -m torch.distributed.launch --nproc_per_node=4 --use_env train_ddp_zinc.py --config './configs/qm9.yml'
  
  ```

## Generating Molecules with 3D-EDiffMG

### De novo generation

```shell
denovo_gen.py --ckpt ./logs/checkpoints/500.pt --save_sdf True --num_samples 10 --sampling_type generalized --w_global_pos 2 --w_global_node 2 --w_local_pos 4 --w_local_node 5
```

### Scaffold-based generation

```shell
python -u scaffold_gen.py --ckpt ./logs/checkpoints/500.pt --save_sdf True --mol_file ./data/scaffold.sdf --keep_index 4 5 10 11 12 13 14 --num_atom 18 --num_samples 100 --sampling_type generalized --w_global_pos 2 --w_global_node 2 --w_local_pos 4 --w_local_node 5
```

To determine the parameter `keep_index`, which selects atom indices within a compound to define the scaffold, you can use the following code:

```python
def get_desired_atom_idx(smiles:str):
    scaffold_mol = Chem.MolFromSmiles(smiles)
    # mol = Chem.AddHs(scaffold_mol)
    for atom in scaffold_mol.GetAtoms():
        atomIdx = atom.GetIdx()
        print(atomIdx, end="\t")
        atomNum = atom.GetAtomicNum()
        print(atomNum, end="\t")
        print(atom.GetSymbol(), end="\t")
        print(atom.GetDegree(), end="\t")
        ExpValence = atom.GetTotalValence()
        print(ExpValence, end="\t")
        print("\n")
```

