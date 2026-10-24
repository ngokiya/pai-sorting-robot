"""
robot.py — Franka Panda の制御（数値IK・関節軌道補間・吸着グリッパ）

吸着グリッパは MuJoCo の weld 等式拘束を ON/OFF することで模擬する
（実機の真空吸着パッドが物体を保持する挙動に相当）。
"""
import numpy as np
import mujoco

DOWN_QUAT = np.array([0.0, 1.0, 0.0, 0.0])  # 手先を真下に向ける姿勢 (w,x,y,z)


class PandaController:
    def __init__(self, model, data, frame_cb=None, fps=30):
        self.m = model
        self.d = data
        self.site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "tcp")
        self.hand_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "hand")
        self.arm_qadr = np.array([model.joint(f"joint{i}").qposadr[0] for i in range(1, 8)])
        self.arm_dofadr = np.array([model.joint(f"joint{i}").dofadr[0] for i in range(1, 8)])
        self.qmin = model.jnt_range[[model.joint(f"joint{i}").id for i in range(1, 8)], 0]
        self.qmax = model.jnt_range[[model.joint(f"joint{i}").id for i in range(1, 8)], 1]
        self.frame_cb = frame_cb
        self.fps = fps
        self.steps_per_frame = int(round(1.0 / fps / model.opt.timestep))
        self._step_count = 0
        self.held = None  # 吸着中の物体名
        self.scratch = mujoco.MjData(model)  # IK 用の計算バッファ
        self.sim_time = 0.0

    # ---------- 低レベル ----------
    @property
    def q(self):
        return self.d.qpos[self.arm_qadr].copy()

    def tcp_pos(self):
        return self.d.site_xpos[self.site_id].copy()

    def step(self, n=1):
        for _ in range(n):
            mujoco.mj_step(self.m, self.d)
            self._step_count += 1
            self.sim_time += self.m.opt.timestep
            if self.frame_cb is not None and self._step_count % self.steps_per_frame == 0:
                self.frame_cb()

    def hold(self, seconds):
        self.step(int(seconds / self.m.opt.timestep))

    # ---------- 逆運動学 ----------
    def ik(self, target_pos, target_quat=DOWN_QUAT, q_init=None, iters=200, tol=1e-4, damping=0.05):
        """減衰最小二乗法 (Damped Least Squares) による位置+姿勢IK。関節角 (7,) を返す。"""
        sd = self.scratch
        sd.qpos[:] = self.d.qpos
        if q_init is not None:
            sd.qpos[self.arm_qadr] = q_init
        jacp = np.zeros((3, self.m.nv))
        jacr = np.zeros((3, self.m.nv))
        quat_err = np.zeros(3)
        site_quat = np.zeros(4)
        neg = np.zeros(4)
        err_quat = np.zeros(4)
        for _ in range(iters):
            mujoco.mj_kinematics(self.m, sd)
            mujoco.mj_comPos(self.m, sd)
            pos_err = target_pos - sd.site_xpos[self.site_id]
            mujoco.mju_mat2Quat(site_quat, sd.site_xmat[self.site_id])
            mujoco.mju_negQuat(neg, site_quat)
            mujoco.mju_mulQuat(err_quat, target_quat, neg)
            mujoco.mju_quat2Vel(quat_err, err_quat, 1.0)
            err = np.concatenate([pos_err, 0.5 * quat_err])
            if np.linalg.norm(pos_err) < tol and np.linalg.norm(quat_err) < 1e-3:
                break
            mujoco.mj_jacSite(self.m, sd, jacp, jacr, self.site_id)
            J = np.vstack([jacp, 0.5 * jacr])[:, self.arm_dofadr]
            dq = J.T @ np.linalg.solve(J @ J.T + damping ** 2 * np.eye(6), err)
            # 関節4(肘)が伸び切らないよう軽いヌルスペース項（可読性のため省略可）
            qn = sd.qpos[self.arm_qadr] + dq
            sd.qpos[self.arm_qadr] = np.clip(qn, self.qmin + 0.02, self.qmax - 0.02)
        return sd.qpos[self.arm_qadr].copy()

    # ---------- 軌道実行 ----------
    def move_joints(self, q_target, duration):
        """現在の関節角から q_target へ、最小躍度(5次)プロファイルで補間しながら位置サーボ指令。"""
        q0 = self.q
        n = max(1, int(duration / self.m.opt.timestep))
        for i in range(1, n + 1):
            s = i / n
            s = 10 * s ** 3 - 15 * s ** 4 + 6 * s ** 5
            self.d.ctrl[:7] = q0 + (q_target - q0) * s
            self.step()

    def move_to(self, pos, duration=1.0, quat=DOWN_QUAT):
        q_t = self.ik(np.asarray(pos, float), quat, q_init=self.q)
        self.move_joints(q_t, duration)
        return np.linalg.norm(self.tcp_pos() - pos)

    def move_linear(self, pos, duration=1.0, quat=DOWN_QUAT, n_way=6):
        """直線経路（複数のIKウェイポイント経由）で移動。降下/上昇に使用。"""
        p0 = self.tcp_pos()
        pos = np.asarray(pos, float)
        for k in range(1, n_way + 1):
            self.move_to(p0 + (pos - p0) * k / n_way, duration / n_way, quat)

    # ---------- 吸着グリッパ ----------
    def suction_on(self, max_dist=0.045):
        """TCP から max_dist 以内で最も近い自由物体を吸着（weld 拘束を有効化）。"""
        tcp = self.tcp_pos()
        best, best_d = None, 1e9
        for i in range(self.m.neq):
            name = mujoco.mj_id2name(self.m, mujoco.mjtObj.mjOBJ_EQUALITY, i)
            if not name or not name.startswith("suction_"):
                continue
            b2 = self.m.eq_obj2id[i]
            # 物体上面の中心との距離で判定
            p = self.d.xpos[b2].copy()
            dist_xy = np.linalg.norm(p[:2] - tcp[:2])
            if dist_xy < best_d:
                best, best_d = (i, b2, name), dist_xy
        if best is None or best_d > max_dist:
            return None
        i, b2, name = best
        # relpose = hand 座標系で見た物体の相対姿勢
        p1, R1 = self.d.xpos[self.hand_id], self.d.xmat[self.hand_id].reshape(3, 3)
        p2 = self.d.xpos[b2]
        rel_p = R1.T @ (p2 - p1)
        q1n = np.zeros(4); rel_q = np.zeros(4)
        mujoco.mju_negQuat(q1n, self.d.xquat[self.hand_id])
        mujoco.mju_mulQuat(rel_q, q1n, self.d.xquat[b2])
        self.m.eq_data[i, 0:3] = 0.0
        self.m.eq_data[i, 3:6] = rel_p
        self.m.eq_data[i, 6:10] = rel_q
        self.d.eq_active[i] = 1
        self.held = name.replace("suction_", "")
        return self.held

    def suction_off(self):
        for i in range(self.m.neq):
            self.d.eq_active[i] = 0
        self.held = None

    # ---------- 高レベル動作 ----------
    def pick(self, xy, z_top, hover=0.15, approach_speed=0.6, approach_offset=0.006):
        """xy 上空 hover から降下 → 吸着 → 上昇。z_top は物体上面の推定高さ。"""
        x, y = xy
        self.move_to([x, y, z_top + hover], 0.9)
        self.move_linear([x, y, z_top + approach_offset], (hover) / approach_speed)
        self.hold(0.15)
        held = self.suction_on()
        self.hold(0.15)
        self.move_linear([x, y, z_top + hover], 0.6)
        return held

    def place(self, xy, obj_height=0.12, hover=0.30, clearance=0.04):
        """ビン上空 hover（把持物体の底面がビン壁・他物体に当たらない高さ）へ移動し、
        物体底面がビン床から clearance になる高さまで降下して解放。obj_height は把持物体の高さ推定値。"""
        x, y = xy
        p0 = self.tcp_pos()
        # 1) 真上に hover まで上げる → 2) 水平に直線移動（他の物体やビン壁を越える高さを保つ）
        self.move_linear([p0[0], p0[1], hover], 0.4, n_way=3)
        dist = np.hypot(x - p0[0], y - p0[1])
        self.move_linear([x, y, hover], max(0.6, dist / 0.6), n_way=max(3, int(dist / 0.08)))
        z_release = max(obj_height + clearance + 0.01, 0.12)
        self.move_linear([x, y, z_release], 0.5, n_way=4)
        self.suction_off()
        self.hold(0.3)
        self.move_linear([x, y, hover], 0.5, n_way=3)
