# Vehicle Specific Power (VSP)

<div align="center">

![Vehicle Dynamics Free Body Diagram](/docs/references/figures/Vehicle-Dynamics-FBD.PNG)  
*Figure 1: Vehicle Dynamics Free Body Diagram (extracted from [Vehicle_Specific_Power.pdf](/docs/references/Vehicle_Specific_Power.pdf)).*

<table>
  <tr>
    <th><b>Force</b></th>
    <th><b>Formula</b></th>
    <th><b>Parameters</b></th>
  </tr>
  <tr>
    <td><b>Inertial Force (Acceleration)</b></td>
    <td>$F_{\text{acc}} = m a$</td>
    <td>$m$ = vehicle mass (kg)  <br> $a$ = vehicle acceleration (m/s²)</td>
  </tr>
  <tr>
    <td><b>Grade (Hill-Climb) Force</b></td>
    <td>$F_{\text{grade}} = mg\sin(\theta)$</td>
    <td>$g$ = gravitational acceleration <br> $\theta$ = road slope angle</td>
  </tr>
  <tr>
    <td><b>Rolling Resistance Force</b></td>
    <td>$F_{\text{rolling}} = mgC_r$</td>
    <td>$C_r$ = rolling resistance coefficient</td>
  </tr>
  <tr>
    <td><b>Aerodynamic Drag Force</b></td>
    <td>$F_{\text{aero}} = \frac{1}{2} \rho_{\text{air}} C_dAv^2$</td>
    <td>$\rho_{\text{air}}$ = air density $1.207$ kg/m³ at 20°C <br> 
        $C_d$ = aerodynamic drag coefficient <br> 
        $A$ = frontal area (m²) <br> 
        $v$ = vehicle speed (m/s)
    </td>
  </tr>
</table>

</div>


**Vehicle Specific Power (VSP)** represents the total power demand **per unit mass** of a vehicle:

$$\text{VSP} = \frac{\frac{d}{dt}\Bigl(E_{\mathrm{kinetic}} + E_{\mathrm{potential}}\Bigr) + F_{\mathrm{rolling}} v + F_{\mathrm{aero}} v}{m}$$

$$ \Rightarrow \text{VSP} \approx v(1 + \epsilon) \cdot a + g \cdot \frac{\text{grade}}{100} + gC_r + \frac{1}{m} \cdot \frac{\rho_{\text{air}}}{2} A C_d v^3$$

where:
- $\upsilon$ = vehicle speed (m/s),
- $E_{\mathrm{kinetic}} = \frac{1}{2} m v^2$,
- $E_{\mathrm{potential}} = mgh$,
- $h =$ elevation,
- $g$ = gravitational acceleration.
- $\epsilon$ = rotational mass factor $\approx$ 0.1

---

**Simplified VSP formula:**

$$VSP = v(1.1a) + 9.81 \frac{r}{100} + 0.132 + 0.000302 v^3$$

where:
- $r$ = road grade,

- simplified constants (approximated):
  - **$0.132$** rolling resistance,
  - **$0.000302 \upsilon^3$** aerodynamic drag.
