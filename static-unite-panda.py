from dm_control import mjcf
import mujoco
import mujoco.viewer
import numpy as np
from scipy.optimize import minimize
import time
import threading
import matplotlib.pyplot as plt
import math

# >>> NUEVO: librería de cinemática inversa (la misma de arm_panda.py)
import mink

# ==========================================================
# 1. ENSAMBLAJE DEL ENTORNO MAESTRO
# ==========================================================
entorno_maestro = mjcf.RootElement()

panda = mjcf.from_path('panda(2).xml')
panda.model = "panda"

brazo_humano = mjcf.from_path('arm26_cvt3.xml')
brazo_humano.model = "humano"

adjunto_panda = entorno_maestro.attach(panda)
adjunto_panda.pos = [0, 0, 0]

adjunto_humano = entorno_maestro.attach(brazo_humano)
adjunto_humano.pos = [0.8, 0, 0]
adjunto_humano.euler = [0, 0, math.pi]

xml_string = entorno_maestro.to_xml_string()
assets_dict = entorno_maestro.get_assets()
model = mujoco.MjModel.from_xml_string(xml_string, assets=assets_dict)
data = mujoco.MjData(model)

# ==========================================================
# 1.5. DESACTIVAR COLISIÓN ENTRE PANDA Y HUMANO
# ==========================================================
# Asignamos los geoms del Panda a un "grupo de contacto" y los del humano a
# otro, de modo que MuJoCo nunca genere contacto entre ambos. Cada geom choca
# con otro solo si (contype_A & conaffinity_B) o (contype_B & conaffinity_A)
# es distinto de cero. Damos al Panda bit 1 y al humano bit 2: como no
# comparten bits, no colisionan entre sí, pero cada uno sigue colisionando
# consigo mismo y con el suelo si correspondiera.
for g in range(model.ngeom):
    name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, g)
    if name is None:
        continue
    if name.startswith("panda/"):
        model.geom_contype[g] = 1
        model.geom_conaffinity[g] = 1
    elif name.startswith("humano/"):
        model.geom_contype[g] = 2
        model.geom_conaffinity[g] = 2
print("Colisión Panda–humano desactivada (grupos de contacto separados).")

# ==========================================================
# 2. IDENTIFICACIÓN DE VARIABLES Y AISLAMIENTO
# ==========================================================
id_shoulder = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "humano/r_shoulder_elev")
id_elbow = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "humano/r_elbow_flex")

# posicion muñeca
id_wrist_site = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "humano/r_radius_styloid_marker")

q_idx_sh = model.jnt_qposadr[id_shoulder]
q_idx_el = model.jnt_qposadr[id_elbow]
v_idx_sh = model.jnt_dofadr[id_shoulder]
v_idx_el = model.jnt_dofadr[id_elbow]

muscle_names = [
    "humano/TRIlong", "humano/TRIlat", "humano/TRImed",
    "humano/BIClong", "humano/BICshort", "humano/BRA"
]
muscle_ids = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, name) for name in muscle_names]
num_muscles = len(muscle_ids)

# ==========================================================
# 2.1. NUEVO: MAPEO DEL BRAZO PANDA PARA LA IK
# ==========================================================
# Efector final del Panda. Este XML NO tiene <site> en el flange, así que
# rastreamos el BODY de la mano. Tras el attach se llama "panda/hand".
EEF_FRAME_NAME = "panda/hand"
EEF_FRAME_TYPE = "body"   # "body" porque no hay site; mink también acepta bodies
id_eef_body = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, EEF_FRAME_NAME)

# Mapeo joint -> actuador del brazo (7 GDL). La IK entrega ÁNGULOS y los
# escribimos como objetivo a los actuadores de POSICIÓN del Panda.
panda_joint_names = [f"panda/joint{i}" for i in range(1, 8)]
panda_act_names = [f"panda/actuator{i}" for i in range(1, 8)]

panda_ctrl_map = []  # lista de (actuator_id, qpos_adr)
for j_name, a_name in zip(panda_joint_names, panda_act_names):
    j_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, j_name)
    a_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, a_name)
    if j_id == -1 or a_id == -1:
        print(f"[ADVERTENCIA] No se encontró {j_name} o {a_name}. Revisa nombres en panda(2).xml.")
        continue
    panda_ctrl_map.append((a_id, model.jnt_qposadr[j_id]))

# --- Diagnóstico de arranque (te ayuda a verificar el modelo sin adivinar) ---
print("\n================ DIAGNÓSTICO DE INTEGRACIÓN ================")
print(f"Efector '{EEF_FRAME_NAME}' ({EEF_FRAME_TYPE}): {'OK (id=%d)' % id_eef_body if id_eef_body != -1 else 'NO ENCONTRADO'}")
print(f"Sitio muñeca: {'OK (id=%d)' % id_wrist_site if id_wrist_site != -1 else 'NO ENCONTRADO'}")
print(f"Joints/actuadores Panda mapeados: {len(panda_ctrl_map)}/7")
# Chequeo simple del tipo de actuador del Panda (posición vs. torque):
if panda_ctrl_map:
    a0 = panda_ctrl_map[0][0]
    is_affine = (model.actuator_biastype[a0] == mujoco.mjtBias.mjBIAS_AFFINE)
    print("Actuadores Panda parecen de POSICIÓN (biastype affine):", bool(is_affine))
    if not is_affine:
        print("  >> OJO: si NO son de posición, escribir ángulos a ctrl no funcionará.")
        print("  >> En ese caso habría que cambiar a control por torque (OSC).")
if id_eef_body == -1:
    print("  >> Lista de bodies disponibles para corregir EEF_FRAME_NAME:")
    for b in range(model.nbody):
        print("     -", mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, b))
print("============================================================\n")

# ==========================================================
# 2.5. INICIALIZACIÓN SEGURA DE LOS MODELOS
# ==========================================================
# A. Asegurar el Panda en su Home Position
panda_joints = ["panda/joint1", "panda/joint2", "panda/joint3", "panda/joint4",
                "panda/joint5", "panda/joint6", "panda/joint7",
                "panda/finger_joint1", "panda/finger_joint2"]
panda_home_qpos = [0, 0, 0, -1.57079, 0, 1.57079, -0.7853, 0.04, 0.04]

for j_name, home_val in zip(panda_joints, panda_home_qpos):
    j_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, j_name)
    if j_id != -1:
        data.qpos[model.jnt_qposadr[j_id]] = home_val

# B. Inicializar el brazo humano ligeramente flectado para evitar el límite 0
data.qpos[q_idx_sh] = 0.2
data.qpos[q_idx_el] = 0.2

mujoco.mj_forward(model, data)

# ==========================================================
# 2.6. CREAR EL "SHADOW DATA" PARA CÁLCULOS SEGUROS
# ==========================================================
calc_data = mujoco.MjData(model)

# ==========================================================
# 2.7. NUEVO: CONFIGURACIÓN DE LA IK (mink)
# ==========================================================
# Parámetros IK (idénticos a arm_panda.py)
IK_SOLVER = "daqp"
IK_DT = 0.02           # paso interno del solver: se subió de 0.005 -> 0.02
                       # (pasos más grandes = converge en menos iteraciones,
                       # el brazo "sigue" a la muñeca más rápido).
POS_THRESHOLD = 5e-4   # más exigente que antes (1e-3) -> el Panda se detiene
                       # más cerca del punto real de la muñeca.
MAX_IK_ITERS = 40      # más iteraciones disponibles para llegar al umbral
                       # más estricto de arriba sin cortar la convergencia.

# Offset respecto a la muñeca, en coordenadas del MUNDO.
# En cero: el Panda apunta EXACTAMENTE a la muñeca (mejor para que "la agarre").
WRIST_OFFSET = np.array([0.0, 0.0, 0.0])

# Largo de las pinzas: distancia desde el origen del body panda/hand hasta la
# PUNTA de los dedos (~0.10 m en el Panda). Sirve para que las pinzas, y no la
# base de la mano, lleguen justo a la muñeca. Si las pinzas quedan cortas, súbelo;
# si se pasan, bájalo.
GRIPPER_LEN = 0.10

# ----------------------------------------------------------------------------
# NUEVO: ORIENTACIÓN DE APROXIMACIÓN BASADA EN EL MARCO DE LA MUÑECA
# ----------------------------------------------------------------------------
# El site "r_radius_styloid_marker" NO tiene quat propio en el XML, así que
# hereda la orientación del body "r_ulna_radius_hand". Ese body SÍ rota de
# forma rígida con el hombro y el codo (es un modelo plano de 2 GDL), por lo
# que su marco de referencia (data.site_xmat) es exactamente lo que
# necesitamos: un eje que "viaja pegado" a la muñeca y cambia si el humano
# levanta o gira el antebrazo.
#
# WRIST_APPROACH_LOCAL_AXIS: qué columna (0=X, 1=Y, 2=Z) del marco de la
# muñeca se usa como dirección "hacia arriba/afuera" por la que debe entrar
# la pinza. Empezamos con Z (tal como lo pediste); si al probarlo en el
# visor el Panda se acerca desde un lado raro, cambia este valor a 0 o 1.
WRIST_APPROACH_LOCAL_AXIS = 2

# Cambia el signo si el brazo se acerca "por debajo" en vez de "por arriba".
WRIST_APPROACH_SIGN = 1.0

# NUEVO: eje LOCAL del marco de la muñeca (distinto de WRIST_APPROACH_LOCAL_AXIS)
# usado para fijar el "arriba" / giro del cabezal, de forma que quede
# perpendicular a la muñeca de verdad (anclado al hueso), y no a un valor
# arbitrario que se arrastra para siempre. Por defecto usamos Y (1), que en
# este modelo es el eje largo del antebrazo (el body "r_ulna_radius_hand" se
# extiende sobre todo en -Y respecto a su padre) -> el cabezal queda alineado
# con el brazo, cruzando la muñeca, en vez de mirando de canto.
# Si en el visor se ve girado 90°, prueba cambiar esto a 0 (X).
WRIST_TWIST_REFERENCE_LOCAL_AXIS = 1

# Distancia (m) a la que se coloca el origen de la mano por encima de la
# muñeca, medida a lo largo del eje de aproximación (además del GRIPPER_LEN,
# que ya compensa el largo físico de los dedos).
WRIST_APPROACH_OFFSET = 0.0

# ORIENTATION_COST: a diferencia de la posición, la orientación NO se exige
# de forma estricta. Le damos un costo "suave" (bajo, no 1.0) para que el
# solver de IK trate de alinear la pinza con el eje de aproximación de la
# muñeca, PERO sin sacrificar la posición ni forzar posturas imposibles.
# En la práctica esto se traduce en una tolerancia angular natural (del
# orden de 20-30°): si alinear perfecto implicaría chocar con el humano o
# una postura extrema, el optimizador prioriza no romper la tarea de
# posición y de postura, y se queda "cerca" de la orientación pedida en vez
# de exacta. mink no soporta un cono de tolerancia explícito, así que este
# es el mecanismo equivalente más simple y robusto.
ORIENTATION_COST = 0.4

configuration = mink.Configuration(model)
configuration.update(data.qpos)

# Tarea principal: el efector del Panda persigue una posición objetivo.
# orientation_cost = 0.0  -> seguimiento SOLO de posición (lo que pediste).
# Para alinear también la orientación del gripper, súbela (ej. 1.0) y fija
# un objetivo de rotación coherente en el bucle.
end_effector_task = mink.FrameTask(
    frame_name=EEF_FRAME_NAME,
    frame_type=EEF_FRAME_TYPE,
    position_cost=1.0,
    orientation_cost=ORIENTATION_COST,
    lm_damping=0.5,  # se bajó de 1.0 -> respuesta más ágil (menos "frenado")
)
# Tarea de postura: regulariza los GDL redundantes del Panda (resuelve el
# nulo de la redundancia 7 GDL -> objetivo 3D). Costo bajo, igual que el original.
posture_task = mink.PostureTask(model=model, cost=1e-2)
posture_task.set_target_from_configuration(configuration)

ik_tasks = {"eef": end_effector_task, "posture": posture_task}


def converge_ik(configuration, tasks, dt, solver, pos_threshold, max_iters):
    """Itera la IK hasta 'max_iters'. Solo verifica POSICIÓN (orientación libre).
    Devuelve True si converge bajo el umbral de posición."""
    for _ in range(max_iters):
        vel = mink.solve_ik(configuration, tasks.values(), dt, solver, damping=1e-3)
        configuration.integrate_inplace(vel, dt)
        err = tasks["eef"].compute_error(configuration)
        if np.linalg.norm(err[:3]) <= pos_threshold:
            return True
    return False

# ==========================================================
# 3. LÓGICA DE CONTROL (Brazo Humano)
# ==========================================================
Kp = 200.0
Kd = 20.0

# Objetivos controlados por teclado (en radianes).
target_elbow_angle = 0.2     # ↑ / ↓
target_shoulder_angle = 0.2  # ← / →

# Límites articulares (del XML): hombro [-1.571, 3.142], codo [0, 2.269].
SHOULDER_MIN, SHOULDER_MAX = -1.50, 3.10
ELBOW_MIN, ELBOW_MAX = 0.05, 2.20

# VELOCIDAD angular al mantener presionada una flecha, en rad/s.
# IMPORTANTE: esto reemplaza al viejo "KEY_STEP por tick". Antes el ángulo
# avanzaba un valor fijo POR VUELTA DEL BUCLE, así que si el bucle iba rápido
# (según cuánto tardara el optimizador de músculos, que es variable) el brazo
# se movía rápido, y si el bucle se atrasaba, lento. Ahora el ángulo avanza
# KEY_RATE_RAD_S * (tiempo real transcurrido), así la velocidad percibida es
# la misma sin importar cuántas iteraciones del bucle ocurran por segundo.
KEY_RATE_RAD_S = 1.2

# Estado de las teclas (qué flechas están presionadas ahora mismo).
keys_pressed = {"up": False, "down": False, "left": False, "right": False}


def _on_press(key):
    try:
        from pynput.keyboard import Key
        if key == Key.up:
            keys_pressed["up"] = True
        elif key == Key.down:
            keys_pressed["down"] = True
        elif key == Key.left:
            keys_pressed["left"] = True
        elif key == Key.right:
            keys_pressed["right"] = True
    except Exception:
        pass


def _on_release(key):
    try:
        from pynput.keyboard import Key
        if key == Key.up:
            keys_pressed["up"] = False
        elif key == Key.down:
            keys_pressed["down"] = False
        elif key == Key.left:
            keys_pressed["left"] = False
        elif key == Key.right:
            keys_pressed["right"] = False
    except Exception:
        pass


def start_keyboard_listener():
    """Inicia el listener de pynput. Si no está instalado, avisa y deja
    el control por teclado desactivado (la simulación igual corre)."""
    try:
        from pynput import keyboard
        listener = keyboard.Listener(on_press=_on_press, on_release=_on_release)
        listener.daemon = True
        listener.start()
        print("Control por teclado ACTIVO:  ↑/↓ = hombro (cinemático)   ←/→ = codo (muscular)")
    except Exception as e:
        print("[AVISO] No se pudo iniciar el teclado (pynput):", e)
        print("        Instala con:  pip install pynput --break-system-packages")


def update_targets_from_keys(dt_wall):
    """Aplica las teclas presionadas a los ángulos objetivo, respetando límites.
    Mapeo:  ↑/↓ = HOMBRO (brazo/bícep adelante-atrás)   ←/→ = CODO (antebrazo sube-baja)
    dt_wall: tiempo REAL (segundos) transcurrido desde la última llamada, para
    que la velocidad de movimiento sea consistente sin importar la frecuencia
    del bucle."""
    global target_elbow_angle, target_shoulder_angle
    step = KEY_RATE_RAD_S * dt_wall
    # ↑ / ↓  ->  hombro
    if keys_pressed["up"]:
        target_shoulder_angle += step
    if keys_pressed["down"]:
        target_shoulder_angle -= step
    # ← / →  ->  codo
    if keys_pressed["right"]:
        target_elbow_angle += step
    if keys_pressed["left"]:
        target_elbow_angle -= step
    target_elbow_angle = float(np.clip(target_elbow_angle, ELBOW_MIN, ELBOW_MAX))
    target_shoulder_angle = float(np.clip(target_shoulder_angle, SHOULDER_MIN, SHOULDER_MAX))


def get_desired_torques(m, d_calc, q_real, qvel_real, q_ref, qvel_ref, qacc_ref):
    qacc_des = qacc_ref + Kp * (q_ref - q_real) + Kd * (qvel_ref - qvel_real)

    # Sincronizamos el shadow data con la realidad antes del cálculo
    d_calc.qpos[:] = q_real
    d_calc.qvel[:] = qvel_real
    d_calc.qacc[:] = qacc_des

    mujoco.mj_inverse(m, d_calc)
    return np.copy(d_calc.qfrc_inverse)


def calculate_muscle_torques(m, d_calc, test_activations):
    # Asumimos que d_calc ya tiene el qpos y qvel correcto del paso actual
    for i, m_id in enumerate(muscle_ids):
        d_calc.ctrl[m_id] = test_activations[i]
        act_idx = m.actuator_actadr[m_id]
        if act_idx != -1:
            d_calc.act[act_idx] = test_activations[i]

    mujoco.mj_fwdActuation(m, d_calc)
    return np.copy(d_calc.qfrc_actuator)


def optimize_activations(m, d_calc, q_real, qvel_real, tau_des, prev_activations, fatigue_state):
    # NOTA IMPORTANTE: de los 6 músculos, solo BIClong y TRIlong cruzan el
    # hombro (se originan en el torso, no en el húmero); los otros 4 solo
    # actúan sobre el codo. Además esos 2 son biarticulares: activarlos para
    # mover el hombro también perturba el codo. Por eso el hombro necesita
    # una prioridad de tracking más alta que el codo, o el optimizador
    # termina "resolviendo" casi todo a favor del codo (que tiene 4 músculos
    # dedicados) y deja el hombro casi sin mover.
    # NUEVO: el hombro ya NO se controla con músculos (ver bloque cinemático
    # en el bucle principal), así que el optimizador solo necesita rastrear
    # el torque del CODO. Se eliminó el término err_sh: ya no tiene sentido,
    # el hombro no depende de lo que hagan los músculos.
    weight_effort = 3.0             # bajado de 10.0: menos freno al esfuerzo muscular
    weight_tracking_elbow = 1.0
    fatigue_penalty_factor = 50.0

    # Preparar el shadow data
    d_calc.qpos[:] = q_real
    d_calc.qvel[:] = qvel_real

    def objective(x):
        tau_muscles = calculate_muscle_torques(m, d_calc, x)
        err_el = (tau_muscles[v_idx_el] - tau_des[v_idx_el])**2
        costo_esfuerzo = (x**2) * (1.0 + fatigue_penalty_factor * fatigue_state)
        effort = np.sum(costo_esfuerzo)
        return (weight_effort * effort) + (weight_tracking_elbow * err_el)

    bounds = [(0.01, 1.0) for _ in range(num_muscles)]

    res = minimize(
        objective,
        prev_activations,
        method='L-BFGS-B',
        bounds=bounds,
        options={'ftol': 1e-4, 'maxiter': 60}  # más iteraciones: el problema
                                                # acoplado hombro-codo necesita
                                                # más margen para converger
    )
    return res.x if res.success else prev_activations

# ==========================================================
# 4. BUCLE PRINCIPAL DE SIMULACIÓN
# ==========================================================
start_keyboard_listener()

time_history = []
activation_history = []

try:
    with mujoco.viewer.launch_passive(model, data) as viewer:
        activations = np.full(num_muscles, 0.01)
        muscle_fatigue = np.zeros(num_muscles)

        fatigue_rate = 0.05
        recovery_rate = 0.02

        current_q_ref = 0.2       # codo suavizado
        current_sh_ref = 0.2      # hombro suavizado

        # Constante de TIEMPO (segundos) del suavizado exponencial, no un
        # factor fijo por tick. Con esto, el ángulo suavizado converge al
        # objetivo en un tiempo real consistente (~3*TAU_SMOOTH para el 95%)
        # sin importar cuántas iteraciones del bucle ocurran por segundo.
        TAU_SMOOTH = 0.15  # más chico = responde más rápido a la tecla

        prev_time = time.time()  # para calcular dt_wall real entre iteraciones

        # Referencia persistente del "giro" (twist) del cabezal del Panda
        # alrededor del eje de aproximación. Se actualiza cada frame por
        # transporte paralelo (ver más abajo). Usar esto, en vez de leer
        # data.xmat del Panda cada frame, evita que el cabezal quede
        # "rotado" o dé saltos bruscos: data.xmat del Panda va con retraso
        # respecto al objetivo (por la dinámica de los actuadores), y usarlo
        # como referencia realimentaba ese retraso y producía giros erráticos.
        wrist_frame_x_ref = None

        print("\nSimulación unificada lista. El Panda seguirá la muñeca del brazo humano.")
        print("Mueve el brazo con las flechas:  ↑/↓ = hombro (cinemático, directo)   ←/→ = codo (muscular)")
        print("(la ventana del visor debe estar enfocada para que el teclado funcione)\n")

        while viewer.is_running():
            step_start = time.time()

            # Tiempo REAL transcurrido desde la iteración anterior (no el
            # timestep fijo de la física). Se usa para que el movimiento por
            # teclado y el suavizado sean consistentes en el tiempo, sin
            # importar si esta vuelta del bucle tardó más o menos por el
            # optimizador de músculos.
            dt_wall = step_start - prev_time
            dt_wall = float(np.clip(dt_wall, 0.0, 0.1))  # evita saltos si el bucle se pausó
            prev_time = step_start

            # Leer teclado y suavizar ambos objetivos
            update_targets_from_keys(dt_wall)
            alpha_eff = 1.0 - np.exp(-dt_wall / TAU_SMOOTH) if dt_wall > 0 else 0.0
            current_q_ref = current_q_ref + alpha_eff * (target_elbow_angle - current_q_ref)
            current_sh_ref = current_sh_ref + alpha_eff * (target_shoulder_angle - current_sh_ref)

            # ==============================================================
            # NUEVO: HOMBRO CINEMÁTICO (sin pasar por el optimizador de
            # músculos). El hombro humano casi no tiene autoridad muscular
            # en este modelo (solo BIClong/TRIlong lo cruzan, y son
            # biarticulares), así que pedirle al optimizador que genere ese
            # torque nunca funciona bien. En vez de eso, igual que con el
            # Panda, fijamos su posición DIRECTAMENTE cada frame: el húmero
            # responde 1:1 a la tecla, sin depender de si el modelo
            # biomecánico puede generar el torque necesario. El codo sigue
            # siendo 100% muscular (sin cambios ahí).
            data.qpos[q_idx_sh] = current_sh_ref
            data.qvel[v_idx_sh] = 0.0
            mujoco.mj_forward(model, data)
            # ==============================================================

            q_real = np.copy(data.qpos)
            qvel_real = np.copy(data.qvel)

            # Referencia perfecta = posición actual para todo el sistema (protege al Panda)
            q_ref = np.copy(q_real)
            qvel_ref = np.zeros(model.nv)
            qacc_ref = np.zeros(model.nv)

            # Sobreescribimos solo el brazo humano (hombro y codo desde teclado)
            q_ref[q_idx_sh] = current_sh_ref
            q_ref[q_idx_el] = current_q_ref

            # --- Control del brazo humano (músculos) ---
            tau_des = get_desired_torques(model, calc_data, q_real, qvel_real, q_ref, qvel_ref, qacc_ref)
            activations = optimize_activations(model, calc_data, q_real, qvel_real, tau_des, activations, muscle_fatigue)

            dt = model.opt.timestep
            muscle_fatigue += fatigue_rate * activations * dt
            muscle_fatigue -= recovery_rate * (1.0 - activations) * dt
            muscle_fatigue = np.clip(muscle_fatigue, 0.0, 1.0)

            if data.time > 0.5:
                time_history.append(data.time)
                activation_history.append(np.copy(activations))

            # Aplicar activación al modelo REAL (humano)
            for i, m_id in enumerate(muscle_ids):
                data.ctrl[m_id] = activations[i]
                act_idx = model.actuator_actadr[m_id]
                if act_idx != -1:
                    data.act[act_idx] = activations[i]

            # ==================================================================
            # NUEVO: CONTROL DEL PANDA POR IK SIGUIENDO LA MUÑECA
            # ==================================================================
            if id_wrist_site != -1 and id_eef_body != -1 and panda_ctrl_map:
                # 1) Posición de la muñeca en el mundo
                wrist_pos = np.copy(data.site_xpos[id_wrist_site]) + WRIST_OFFSET

                # 2) Sincronizar la configuración de la IK con el estado REAL.
                configuration.update(data.qpos)

                # 3) NUEVO: marco de referencia de la MUÑECA (no del Panda).
                #    Este marco rota junto con el hombro/codo del humano, así
                #    que si el humano levanta o gira el antebrazo, el eje de
                #    aproximación cambia con él.
                R_wrist = np.copy(data.site_xmat[id_wrist_site]).reshape(3, 3)
                approach_dir = WRIST_APPROACH_SIGN * R_wrist[:, WRIST_APPROACH_LOCAL_AXIS]
                approach_dir = approach_dir / np.linalg.norm(approach_dir)

                # 4) Ejes X/Y del objetivo ANCLADOS al marco real de la
                #    muñeca (no a un valor arbitrario ni al estado con
                #    retraso del Panda). Usamos otro eje del propio
                #    site_xmat de la muñeca como referencia de "arriba", así
                #    el cabezal queda físicamente perpendicular a la muñeca
                #    y se autocorrige solo en cada frame si el brazo gira.
                z_axis = -approach_dir
                ref_world = R_wrist[:, WRIST_TWIST_REFERENCE_LOCAL_AXIS]
                x_axis = ref_world - np.dot(ref_world, z_axis) * z_axis
                norm_x = np.linalg.norm(x_axis)

                if norm_x < 1e-3:
                    # Caso degenerado: el eje de referencia quedó casi
                    # paralelo al eje de aproximación (ambos ejes de la
                    # muñeca casi alineados). Usamos la continuidad del
                    # frame anterior solo como respaldo puntual, no como
                    # regla general.
                    if wrist_frame_x_ref is not None:
                        x_axis = wrist_frame_x_ref - np.dot(wrist_frame_x_ref, z_axis) * z_axis
                        norm_x = np.linalg.norm(x_axis)
                    if norm_x < 1e-6:
                        seed = np.array([0.0, 0.0, 1.0])
                        x_axis = seed - np.dot(seed, z_axis) * z_axis
                        norm_x = np.linalg.norm(x_axis)

                x_axis = x_axis / norm_x
                y_axis = np.cross(z_axis, x_axis)
                R_target = np.column_stack([x_axis, y_axis, z_axis])
                wrist_frame_x_ref = x_axis  # respaldo para el caso degenerado

                # 5) Compensación TCP: la IK lleva el ORIGEN de panda/hand al
                #    objetivo, pero las pinzas están GRIPPER_LEN metros más
                #    adelante (a lo largo de +Z local de la mano). Para que
                #    las PINZAS lleguen a la muñeca entrando por el eje de
                #    aproximación, retrocedemos el origen de la mano esa
                #    distancia (más el offset opcional "por encima").
                coord_muneca = (
                    wrist_pos
                    + (GRIPPER_LEN + WRIST_APPROACH_OFFSET) * approach_dir
                )

                T_target = mink.SE3.from_rotation_and_translation(
                    mink.SO3.from_matrix(R_target),
                    coord_muneca,
                )
                end_effector_task.set_target(T_target)

                # 4) Resolver IK
                converge_ik(
                    configuration, ik_tasks, IK_DT, IK_SOLVER,
                    POS_THRESHOLD, MAX_IK_ITERS,
                )

                # 5) Escribir SOLO los actuadores del Panda (los músculos quedan intactos)
                q_sol = configuration.q
                for a_id, q_adr in panda_ctrl_map:
                    data.ctrl[a_id] = q_sol[q_adr]

            # ==================================================================

            mujoco.mj_step(model, data)

            # Reforzar el hombro cinemático tras el paso de física: aunque
            # ya se fijó arriba, mj_step integra un paso más usando la
            # aceleración de ese instante (gravedad, músculos biarticulares,
            # etc.), lo que podría moverlo una fracción mínima. Se corrige
            # aquí para que el visor nunca muestre ni un frame de "flote".
            data.qpos[q_idx_sh] = current_sh_ref
            data.qvel[v_idx_sh] = 0.0
            mujoco.mj_forward(model, data)

            viewer.sync()

            time_until_next_step = model.opt.timestep - (time.time() - step_start)
            if time_until_next_step > 0:
                time.sleep(time_until_next_step)

except KeyboardInterrupt:
    print("\nSimulación detenida manualmente (Ctrl+C).")

# ==========================================================
# 5. GENERACIÓN DEL GRÁFICO
# ==========================================================
print("\nProcesando gráfico...")

activation_history = np.array(activation_history)
plt.figure(figsize=(10, 6))

for i in range(num_muscles):
    clean_name = muscle_names[i].replace("humano/", "")
    plt.plot(time_history, activation_history[:, i], label=clean_name, linewidth=2)

plt.xlabel('Pasos de Simulación (Tiempo)')
plt.ylabel('Nivel de Activación (0 a 1)')
angulo_grados = int(np.round(np.rad2deg(target_elbow_angle)))
plt.title(f'Activaciones Musculares (Brazo) - Objetivo Final: {angulo_grados}°')
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.grid(True, linestyle='-', alpha=0.3)
plt.tight_layout()
plt.savefig("grafico_desgaste_unificado.png", dpi=300)
print("Archivo guardado exitosamente.")
plt.show()
