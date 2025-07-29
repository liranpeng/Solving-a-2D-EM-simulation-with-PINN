# 🌀 2D Electromagnetic Simulation with PINN (Helmholtz Equation)

This repository demonstrates how to solve a **2D electromagnetic (EM) Helmholtz equation** using a **Physics-Informed Neural Network (PINN)** based on the **SIREN** (sinusoidal representation network) architecture. 

The code implements a PINN to approximate solutions for:
- ✅ Isotropic media
- ✅ Optional anisotropic extensions (VTI, TTI)
- ✅ Open boundary conditions with **PML** (perfectly matched layer) absorption
- ✅ Configurable geometries: **circle**, **square**, and **topography**

The approach solves the steady-state Helmholtz equation in frequency domain:

\[ (-\nabla^2 - \epsilon \omega^2) E_z = -i \omega J_z \]

Where:
- **E_z**: Complex out-of-plane electric field amplitude
- **ε**: Permittivity (can vary spatially)
- **ω**: Angular frequency
- **J_z**: Source term (Gaussian source used here instead of a delta function)
- **PML**: Absorption layer near boundaries to mimic open-domain conditions

---

## 📂 Repository Structure

- `notebooks/`
  - **`helmholtz_circle.ipynb`** → Helmholtz PINN with a dielectric **circle** in the center
  - **`helmholtz_square.ipynb`** → Helmholtz PINN with a dielectric **square**
  - **`helmholtz_topography.ipynb`** → Helmholtz PINN with sinusoidal **topography**

Each notebook allows switching between modes and contains visualization code for the **real** and **imaginary** components of \(E_z\), with contours for:
- **Source (J_z)** → White contours
- **Permittivity distribution (ε)** → Black contours

---

## ✨ Features

✅ **Physics-Informed Neural Network (PINN)**: Uses PDE residual as loss  
✅ **SIREN layers**: Enables high-frequency function learning with sinusoidal activations  
✅ **Gaussian Source**: Smooth excitation instead of single-pixel delta → easier for PINN to capture  
✅ **Absorbing Boundary Layer (PML)**: Suppresses reflections at domain boundaries  
✅ **Multiple Scenarios**: Circle, square, or sinusoidal topography geometry  
✅ **Real & Imaginary Solutions**: Visualize propagating waves and complex field patterns  

---

## 📊 Example Results

The code produces plots like these:

- **Permittivity map** (ε)
- **Source J_z** distribution
- **Real(E_z)** and **Imag(E_z)** with contour overlays

---

## 🚀 Getting Started

### 1️⃣ Clone the repository
```bash
git clone https://github.com/yourusername/2D-Helmholtz-PINN.git
cd 2D-Helmholtz-PINN
```

### 2️⃣ Install dependencies
```bash
pip install numpy torch matplotlib
```

### 3️⃣ Run the notebooks
Open any of the provided notebooks and run:
- `helmholtz_circle.ipynb`
- `helmholtz_square.ipynb`
- `helmholtz_topography.ipynb`

Each notebook:
- Defines the geometry and permittivity distribution
- Sets up the Gaussian source and PML
- Builds and trains the PINN
- Visualizes **Real(E_z)** and **Imag(E_z)**

---

## ⚙️ Key Parameters

You can tune the following in each notebook:
- **`case_type`**: `"circle"`, `"square"`, or `"topography"`
- **`omega`**: Angular frequency of the wave
- **`layers`**: PINN architecture depth/width
- **`pml_width`**: Thickness of absorption layer
- **`ps`**: Source (Gaussian width and amplitude)

---

## 📚 Reference

The code is inspired by Helmholtz equation PINN formulations and uses the **SIREN** architecture:  
[Sitzmann et al., 2020 – Implicit Neural Representations with Periodic Activation Functions](https://arxiv.org/abs/2006.09661)

---

## 🏆 Notes & Future Work

✅ Current version uses a **Gaussian source** for stability  
✅ Frequency slightly tuned for easier convergence  
✅ Includes **PML absorption** (simple damping function)  

🔜 Future improvements:
- Extend to **3D EM fields**
- Add **multi-frequency training**
- Compare against **finite-difference Helmholtz solvers** for validation

---

## 📜 License

MIT License – Feel free to modify and use for research and learning.

---
🚀 **Enjoy solving 2D EM Helmholtz problems with neural networks!**
