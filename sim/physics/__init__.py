"""The engineering equations, as small pure functions (numbers / numpy in, numbers out).

No MuJoCo, no scenes, no file I/O except reading part data. Each module is one domain and
states its equations in its docstring:

    propeller   thrust, power, torque, inertia; operating point of a motor + prop
    dc_motor    brushless motor (KV, R, I0), brushed gearmotor torque-speed line, lag
    battery     open-circuit voltage vs state of charge, pack terminal voltage under load
    beam        tube sections, cantilever deflection, first bending mode, stress
    vehicle     wheel torques (rolling, slope, traction), tip-over, steering clearance
    aero        air drag, the analytic wind field
    vibration   transmissibility of mounts and structures, anti-alias filters
    noise       Gauss-Markov processes, measurement delay lines

The same functions serve sizing (sim/design), simulation (plants, sensors) and the C++
core's reference tests, so these numbers cannot drift apart.
"""
