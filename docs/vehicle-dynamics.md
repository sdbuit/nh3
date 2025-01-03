# Vehicle Dynamics

## Vehicle Specific Power (VSP)

**Vehicle Specific Power (VSP)** is a measure used in transportation engineering and emission modeling to represent the power demand per unit mass of a vehicle. A general form is:

$$
\text{VSP} 
= \frac{\text{Power}}{m}
= \frac{\frac{d}{dt}\Bigl(E_{\mathrm{kinetic}} + E_{\mathrm{potential}}\Bigr) 
   + F_{\mathrm{rolling}} \cdot v 
   + F_{\mathrm{aerodynamic}} \cdot v 
   + F_{\mathrm{internal\ friction}} \cdot v}{m},
$$

where:
- $( m )$ is the vehicle mass,  
- $( v )$ is the vehicle speed,  
- $( E_{\mathrm{kinetic}} = \tfrac{1}{2} m v^2 )$,  
- $( E_{\mathrm{potential}} = mgh )$,  
- $( g )$ is gravitational acceleration,  
- $( F_{\mathrm{rolling}}, F_{\mathrm{aerodynamic}}, F_{\mathrm{internal\ friction}} )$ are rolling, aerodynamic, and internal friction forces.

An **approximate** model for VSP often appears as:

$$\text{VSP} \approx
v\,a \,(1 + \epsilon_i)
+ g \,\text{grade}\,v
+ g \,C_{R}\,v
+ \tfrac{1}{2}\,\rho_{a}\,C_{D} \,\frac{A}{m}\,(v + v_{w})^2 \,v
+ C_{\mathrm{if}}\,v,$$

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

VSP can also be expressed via the **tractive force** at the wheels:

$$\text{VSP} 
= \frac{F_{\mathrm{traction}} \cdot v}{m},$$

$$F_{\mathrm{traction}}
= m\,a
  + F_{\mathrm{rolling}}
  + F_{\mathrm{aero}}
  + F_{\mathrm{grade}},$$

leading to:

$$\text{VSP}
= \frac{\bigl(m \, a 
           + F_{\mathrm{rolling}} 
           + F_{\mathrm{aero}} 
           + F_{\mathrm{grade}}\bigr)\,v}{m}$$

**1. Inertial Force (Acceleration)**  
   $$F_{\mathrm{inertial}} = m\,a$$

**2. Rolling Resistance**  
   $$F_{\mathrm{rolling}} 
   = C_{r}\,m\,g\,\cos(\theta),$$
   where $(C_{r})$ is the rolling resistance coefficient, and $(\theta)$ is the road grade angle (radians).

**3. Aerodynamic Drag**  
   $$
   F_{\mathrm{aero}}
   = \tfrac{1}{2}\,\rho\,C_{d}\,A\,v^2,
   $$
   where $(\rho)$ is air density, $(C_{d})$ is drag coefficient, and $(A)$ is frontal area.

**4. Gravitational (Grade) Force**
   $$F_{\mathrm{grade}}
   = m\,g\,\sin(\theta)$$

A final form for VSP after combining:

$$\text{VSP} 
= \frac{\Bigl(m\,a 
             + C_{r}\,m\,g\,\cos(\theta) 
             + \tfrac{1}{2}\,\rho\,C_{d}\,A\,v^2 
             + m\,g\,\sin(\theta)\Bigr)\,v}{m},$$

and simplifying yields:

$$\text{VSP} 
= a\,v
  + C_{r}\,g\,\cos(\theta)\,v
  + \frac{\tfrac{1}{2}\,\rho\,C_{d}\,A\,v^3}{m}
  + g\,\sin(\theta)\,v.$$
