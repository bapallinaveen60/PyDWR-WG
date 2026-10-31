# Mathematical Formulation and Physical Constraints

## 1. Polarimetric Radar Observables
At meteorological radar wavelengths (S-band $\lambda \approx 10.7$ cm), the principal polarimetric observables in STAR mode (simultaneous transmit and receive) are defined as:

$$Z_H = \frac{\pi^5 |K_w|^2}{\lambda^4} \int |s_{hh}(\pi)|^2 N(D) dD \quad [\text{mm}^6 \text{ m}^{-3}]$$

$$Z_{DR} = 10 \log_{10} \left( \frac{\int |s_{vv}(\pi)|^2 N(D) dD}{\int |s_{hh}(\pi)|^2 N(D) dD} \right) = Z_H - Z_V \quad [\text{dB}]$$

$$\rho_{HV}(0) = \frac{\int s_{vv}^*(\pi) s_{hh}(\pi) N(D) dD}{\sqrt{\int |s_{hh}(\pi)|^2 N(D) dD \cdot \int |s_{vv}(\pi)|^2 N(D) dD}}$$

$$\Phi_{DP}(r) = \Phi_{DP}(0) + 2 \int_0^r K_{DP}(s) ds + \delta_{CO}(r) \quad [^\circ]$$

$$K_{DP} = \frac{\pi}{180 \lambda} \text{Re} \left[ \int (f_{hh}(0) - f_{vv}(0)) N(D) dD \right] \approx \frac{1}{2} \frac{\partial \Phi_{DP}}{\partial r} \quad [^\circ \text{ km}^{-1}]$$

where $|K_w|^2 \approx 0.93$ is the dielectric factor of liquid water, $N(D)$ is the drop size distribution across diameter $D$, and $\delta_{CO}$ is the backscatter differential phase shift.

## 2. Invariance of Differential Phase to Power Blockage
When an orographic ridge intercepts a fraction of the radar beam ($BBF > 0$), received backscatter powers $P_h$ and $P_v$ drop by:

$$\Delta P = 10 \log_{10}(1 - BBF) \quad [\text{dB}]$$

introducing severe negative biases in $Z_H$ and $Z_{DR}$. Conversely, the differential propagation phase $\Phi_{DP}$ measures the differential velocity of horizontally and vertically polarized electric fields through anisotropic hydrometeors:

$$\left(\frac{\partial \Phi_{DP}}{\partial r}\right)_{\text{blocked}} = \left(\frac{\partial \Phi_{DP}}{\partial r}\right)_{\text{unblocked}} = 2 K_{DP}$$

Therefore, the integrated downstream phase accumulation records the true hydrometeor water path, regardless of power attenuation behind mountain crests.

## 3. Composite Loss Function
PTA-InNet is optimized using a four-component loss function:

$$\mathcal{L}_{\text{total}} = \lambda_1 \mathcal{L}_{\text{seg}} + \lambda_2 \mathcal{L}_{\text{rec}} + \lambda_3 \mathcal{L}_{\text{grad}} + \lambda_4 \mathcal{L}_{\text{phys}}$$

### 3.1 Clutter Segmentation Loss
$$\mathcal{L}_{\text{seg}} = \mathcal{L}_{\text{Focal}}(\mathbf{M}, \mathbf{\hat{M}}) + \mathcal{L}_{\text{Jaccard}}(\mathbf{M}, \mathbf{\hat{M}})$$

with $\gamma = 2.0$ and $\alpha = 0.25$.

### 3.2 Huber Reflectivity Reconstruction Loss
$$\mathcal{L}_{\text{rec}} = \frac{1}{|\Omega_{\text{valid}}|} \sum_{(r, \theta) \in \Omega_{\text{valid}}} \text{Huber}(Z_H^{\text{true}}(r, \theta) - \hat{Z}_H^{\text{clean}}(r, \theta); \delta=1.0)$$

### 3.3 Spatial Gradient Regularization
$$\mathcal{L}_{\text{grad}} = \frac{1}{|\Omega|} \sum_{(r, \theta)} \|\nabla \hat{Z}_H^{\text{clean}}(r, \theta) - \nabla Z_H^{\text{true}}(r, \theta)\|_1$$

### 3.4 Physical Propagation Phase Constraint
$$\mathcal{L}_{\text{phys}} = \frac{1}{N_{\text{rays}}} \sum_\theta \max\left(0, [\Phi_{DP}(r_2, \theta) - \Phi_{DP}(r_1, \theta)] - 2 \int_{r_1}^{r_2} \psi(\hat{Z}_H^{\text{clean}}(r, \theta), Z_{DR}(r, \theta)) dr - \epsilon\right)$$

where $\psi(Z_H, Z_{DR})$ is the empirical mapping to expected $K_{DP}$, enforcing microphysical conservation laws across blind zones.
