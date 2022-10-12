# GraphMVM: Multi-View Mask Learning for Self-Supervised Graph Representation

Self-supervised graph representation learning is a key technique for graph structured data processing, especially for Web-generated graph that do not have qualified labelling information.
## Dependencies

```python
pip install -r requirements.txt
```

## Usage

You can use the following command, and the parameters are given

```python
python main.py --dataset Cora
```

The `--dataset` argument should be one of [Cora, CiteSeer, PubMed].

## Reference link

The code refers to the following two papers. Thank them very much for their open source work.

[Deep Graph Contrastive Representation Learning(GRACE)](https://github.com/CRIPAC-DIG/GRACE)

[GraphMAE: Self-Supervised Masked Graph Autoencoders](https://github.com/THUDM/GraphMAE)