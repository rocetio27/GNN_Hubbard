# GNN_Hubbard
Machine learning for correlated electrons: solving Hubbard system with graph neural networks
<p align="center">
  <img src="figures/Figure_1.svg" width="480"
       alt="Relative-error heat-map of GNN vs exact diagonalization"/>
</p>

This repository provides a complete, end-to-end pipeline for the **1-D Hubbard model with \(N\) sites** (periodic boundary conditions, two spin flavours, **half-filling**):

1. **Train a Graph Neural Network (GNN)** on exact data generated for a range of Hubbard parameters \((U,t)\).
2. **Predict the ground-state energy** of the \(N\)-site chain for arbitrary \((U,t)\) using the trained GNN.
3. **Assess the model** by comparing its predictions with **exact diagonalisation** and plotting a heat-map of the **relative error** across the \((U,t)\) plane.

The repository thus covers everything from data creation and GNN training to quantitative evaluation and visualisation of the results.
