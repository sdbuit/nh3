# Vehicle Dynamics

## Vehicle Specific Power (VSP)

**Vehicle Specific Power (VSP)** is a measure used in transportation engineering and emission modeling to represent the power demand per unit mass of a vehicle. A general form is:

<div align="center" style="text-align: center;">

$$\text{VSP}=\frac{\text{Power}}{m}=\frac{\frac{d}{dt}\Bigl(E_{\mathrm{kinetic}} + E_{\mathrm{potential}}\Bigr)+F_{\mathrm{rolling}} \cdot v+F_{\mathrm{aerodynamic}} \cdot v +F_{\mathrm{internal\ friction}} \cdot v}{m},$$

</div>

where:
- $( m )$ is the vehicle mass,  
- $( v )$ is the vehicle speed,  
- $( E_{\mathrm{kinetic}} = \tfrac{1}{2} m v^2 )$,  
- $( E_{\mathrm{potential}} = mgh )$,  
- $( g )$ is gravitational acceleration,  
- $F_{\mathrm{rolling}}, F_{\mathrm{aerodynamic}},F_{\mathrm{internal\ friction}}$ are rolling, aerodynamic, and internal friction forces.

An **approximate** model for VSP often appears as:

$$\text{VSP} \approx va \cdot (1 + \epsilon_i) + g \cdot \text{grade} \cdot v + g \cdot C_{R} \cdot v + \tfrac{1}{2} \cdot \rho_{a} \cdotp C_{D} \cdot \left ( \frac{A}{m} \right ) \cdot (v + v_{w})^2 \cdot v + C_{\mathrm{if}} \cdot v,$$

where:
- $( a )$ is acceleration,  
- $( \epsilon_i )$ is a correction factor,  
- $(\text{grade} = \tfrac{\Delta h}{\Delta x}$),  
- $(C_{R})$ is the rolling-resistance coefficient,  
- $(\rho_{a})$ is air density,  
- $(C_{D})$ is the aerodynamic drag coefficient,  
- $(A)$ is frontal area,  
- $(v_{w})$ is headwind speed,  
- $(C_{\mathrm{if}})$ is internal friction coefficient.

---

<br>

VSP can also be expressed via the **tractive force** at the wheels:

$$\text{VSP}=\frac{F_{\mathrm{traction}}\cdot v}{m}$$

$$F_{\mathrm{traction}}=ma+F_{\mathrm{rolling}}+F_{\mathrm{aero}}+F_{\mathrm{grade}},$$

leading to:

$$\text{VSP} = \frac{\bigl(ma + F_{\mathrm{rolling}}+F_{\mathrm{aero}} + F_{\mathrm{grade}}\bigr) \cdot v}{m}$$

<br>

**1. Inertial Force (Acceleration)**
   
   $$F_{\mathrm{inertial}} = ma$$

**2. Rolling Resistance**  
   
   $$F_{\mathrm{rolling}}=C_{r} \ mg \ \cos(\theta),$$
   
   where $(C_{r})$ is the rolling resistance coefficient, and $(\theta)$ is the road grade angle (radians).

**3. Aerodynamic Drag**  
   $$F_{\mathrm{aero}}= \left( \tfrac{1}{2} \right) \cdot \rho \ C_{d} \ A \ v^2$$
   
   where $(\rho)$ is air density, $(C_{d})$ is coefficient 
   and $(A)$ is frontal area.

**4. Gravitational (Grade) Force**

   $$F_{\mathrm{grade}}=m\,g\,\sin(\theta)$$

A final form for VSP after combining:

$$\text{VSP}=\frac{\Bigl(ma+C_{r} \cdot mg \cdot \cos(\theta)+ \left(\frac{1}{2} \right) \rho C_{d}Av^2+mg\sin(\theta)\Bigr)\cdot v}{m},$$

and simplifying yields:

$$\begin{equation} \text{VSP}=a\cdot{v}+C_{r}\cdot g \cdot \cos(\theta)\cdot v+\frac{(\tfrac{1}{2}) \cdot \rho \cdot C_{d} \cdot A \cdot{v^3}}{m}+g \cdot \sin(\theta)\cdot{v}\end{equation}$$
