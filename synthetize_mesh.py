import os
import numpy as np
import trimesh
import random

import trimesh.creation
import trimesh.exchange
import trimesh.exchange.obj

# -------------------------
# 1. 複数トポロジーの mesh を準備
# -------------------------
def create_base_meshes():
	"""
	球、トーラス、ボックスなど、いくつかのトポロジーを持つmeshを返す。
	必要に応じて追加・修正可能。
	"""
	sphere = trimesh.creation.icosphere(subdivisions=2, radius=1.0)
	annulus = trimesh.creation.annulus(r_min=0.5, r_max=1.0, height=0.2)
	torus = trimesh.creation.torus(major_radius=1.0, minor_radius=0.3)
	capsule = trimesh.creation.capsule(height=2.0, radius=0.5)
	cone = trimesh.creation.cone(radius=1.0, height=2.0)
	cylinder = trimesh.creation.cylinder(radius=1.0, height=2.0)
	box = trimesh.creation.box(extents=(1.0, 1.0, 1.0))
	return [sphere, torus, box, annulus, capsule, cone, cylinder]

# -------------------------
# ユーティリティ: ランダム変形
# -------------------------
def apply_random_transform(mesh: trimesh.Trimesh,
						   translation_range=1.0,
						   scale_range=(0.8, 1.2),
						   rotation_degrees=30,
						   shear_range=0.2):
	"""
	mesh に対して以下のランダムなアフィン変換をまとめて適用する:
	 - ランダム平行移動 (±translation_range)
	 - ランダム回転 (±rotation_degrees 度)
	 - ランダムスケーリング (scale_range)
	 - ランダムシアー (±shear_range)
	これにより対称ではない変形も加わり、オブジェクト間の違いが大きくなる。
	"""

	# ---------------------------
	# 1) 平行移動行列 (Translation)
	# ---------------------------
	T = np.eye(4)
	T[0, 3] = random.uniform(-translation_range, translation_range)
	T[1, 3] = random.uniform(-translation_range, translation_range)
	T[2, 3] = random.uniform(-translation_range, translation_range)

	# ---------------------------
	# 2) 回転行列 (Rotation)
	# ---------------------------
	axis = np.random.randn(3)
	axis /= np.linalg.norm(axis)  # 回転軸を正規化
	angle = np.radians(random.uniform(-rotation_degrees, rotation_degrees))
	R = trimesh.transformations.rotation_matrix(angle, axis)

	# ---------------------------
	# 3) スケーリング行列 (Scale)
	# ---------------------------
	scale_factor = random.uniform(scale_range[0], scale_range[1])
	S = np.eye(4)
	np.fill_diagonal(S, scale_factor)

	# ---------------------------
	# 4) シアー行列 (Shear)
	#    ここでは x→y, x→z, y→z の3つのせん断要素をランダム生成
	# ---------------------------
	H = np.eye(4)
	H[0, 1] = random.uniform(-shear_range, shear_range)  # X軸方向を Yでシアー
	H[0, 2] = random.uniform(-shear_range, shear_range)  # X軸方向を Zでシアー
	H[1, 2] = random.uniform(-shear_range, shear_range)  # Y軸方向を Zでシアー

	# 変換の順序については用途によって変わるが、ここでは
	#   M = T * R * H * S
	# の順に合成している。(行列積の順番に注意)
	# p_world = M * p_local
	# → mesh.apply_transform(M)
	M = T @ R @ H @ S

	mesh.apply_transform(M)
	return mesh

# ------------------------------------------------
# バウンディングスフィアを計算するユーティリティ
# ------------------------------------------------
def compute_bounding_sphere(mesh: trimesh.Trimesh):
	"""
	メッシュのバウンディングスフィアを(中心座標, 半径) で返す。
	ここでは「重心を中心とし、そこから最大距離を半径とする」簡易的な定義。
	"""
	center = mesh.center_mass  # or mesh.centroid
	# 頂点から中心への最大距離を半径とする
	radius = np.max(np.linalg.norm(mesh.vertices - center, axis=1))
	return center, radius

# ------------------------------------------------
# 3) クラスメッシュを生成
#    → 必ずオーバーラップが生じるようにステップごとに Union
# ------------------------------------------------
def generate_class_mesh(mesh_list, num_meshes=4):
	"""
	- mesh_list の中からランダムにメッシュを取り出し、1つ目をベースとする。
	- 2つ目以降は、current_union の重心に向かって移動させ、必ずオーバーラップが生じるように配置して Union する。
	- 最終的に current_union を返す。
	"""

	# 1) まず1つ目のメッシュをランダムに選んで current_union とする
	current_union = random.choice(mesh_list).copy()
	current_union = apply_random_transform(current_union, translation_range=0.5)

	# 2) Union する数が1つだけなら、そのまま返す
	if num_meshes == 1:
		return current_union

	for _ in range(num_meshes-1):
		# 次のメッシュをランダム選択してコピー
		new_mesh = random.choice(mesh_list).copy()
		new_mesh = apply_random_transform(new_mesh, translation_range=0.5)

		# それぞれのバウンディングスフィア (中心, 半径)
		cu_center, cu_radius = compute_bounding_sphere(current_union)
		nm_center, nm_radius = compute_bounding_sphere(new_mesh)

		# 現在の union の中心 cu_center と new_mesh の中心 nm_center を結ぶベクトル
		direction = cu_center - nm_center
		dist = np.linalg.norm(direction)

		if dist < 1e-9:
			# 万が一ほぼ同一点なら、何らかの適当な方向を与える
			direction = np.array([1.0, 0.0, 0.0])
			dist = 1.0

		# unit vector に正規化
		direction /= dist

		# 「両者のバウンディングスフィアがオーバーラップする」ためには
		#    center間距離 < (cu_radius + nm_radius)
		# となる必要がある。
		# ランダムに factor を決めて、スフィア半径合計のうち何割の距離にするか決める。
		factor = random.uniform(0.6, 1.0)  # 0.6〜1.0の範囲で設定 (完全貫通～ぎりぎり重なる程度)
		desired_dist = (cu_radius + nm_radius) * factor

		# 現在 dist があるので、移動量は (desired_dist - dist)
		move_amount = desired_dist - dist

		# move_amount が正なら new_mesh の中心を current_union の中心へ近づける形になる
		# 負になると、逆に離れる形になる。
		# 今回は必ずオーバーラップしたいので、factor を 1 未満にすれば確実に正値になることが多い。
		# ただし factor=1 付近だと "ぎりぎり重なる" 程度になることがあります。
		translation_vec = direction * move_amount

		# 4x4 平行移動行列を作って適用
		T = np.eye(4)
		T[:3, 3] = translation_vec
		new_mesh.apply_transform(T)

		# 最後に current_union と new_mesh を Union
		current_union = trimesh.boolean.union([current_union, new_mesh], engine='manifold')

	return current_union

# -------------------------
# 4. クラスメッシュから複数のオブジェクトメッシュを生成
# -------------------------
def generate_objects_from_class(class_mesh: trimesh.Trimesh,
								num_objects=3):
	"""
	与えられたクラスメッシュにランダム変形を加え、
	num_objects 個のオブジェクト用メッシュを作成して返す。
	"""
	object_meshes = []
	for _ in range(num_objects):
		mesh_copy = class_mesh.copy()
		mesh_copy = apply_random_transform(mesh_copy,
							   scale_range=(0.9, 1.1),
							   rotation_degrees=15,
							   shear_range=0.4)
		object_meshes.append(mesh_copy)
	return object_meshes

# -------------------------
# ユーティリティ: メッシュの正規化関数 (AABB方式)
# -------------------------
def normalize_mesh_AABB(mesh: trimesh.Trimesh):
	"""
	バウンディングボックスの最大辺が 1 になるようにスケーリングし、
	その後メッシュ中心が原点になるように平行移動する。
	"""
	# min, max 座標
	bounds = mesh.bounds  # shape: (2, 3) [min, max]
	size = bounds[1] - bounds[0]
	max_dim = np.max(size)

	# ゼロ除算の可能性を回避
	if max_dim < 1e-9:
		return  # 極端に小さい場合はスキップ

	# スケーリング行列
	scale_factor = 1.0 / max_dim
	S = np.eye(4)
	np.fill_diagonal(S, scale_factor)

	# バウンディングボックス中心
	center = (bounds[0] + bounds[1]) / 2.0
	T = np.eye(4)
	T[:3, 3] = -center

	# スケール→平行移動の順で適用（順番は用途によって調整可能）
	transform = trimesh.transformations.concatenate_matrices(S, T)
	mesh.apply_transform(transform)

# -------------------------
# 5. データセット構築
# -------------------------
def build_dataset(base_dir='dataset',
				  num_classes=2,
				  objects_per_class=3):
	"""
	データセットを以下の構成で作成する:
	dataset
	├─class_0
	│  ├─object_0
	│  │  └─models
	│  │      └─model_normalized.obj
	│  ├─object_1
	│  ...
	├─class_1
	   ├─object_0
	   ├─object_1
	   ...
	"""
	# 1. 基本形状メッシュを準備
	base_meshes = create_base_meshes()

	# データセットディレクトリを作成
	os.makedirs(base_dir, exist_ok=True)

	for c_idx in range(num_classes):
		# クラスフォルダ
		class_dir = os.path.join(base_dir, f"class_{c_idx}")
		os.makedirs(class_dir, exist_ok=True)

		# 3. クラスメッシュを生成 (球やトーラスなどを Union)
		class_mesh = generate_class_mesh(base_meshes, num_meshes=2 + random.randint(0,1))
		
		# 必要に応じてクラスメッシュ自体も正規化する場合
		normalize_mesh_AABB(class_mesh)

		# 4. クラスメッシュからオブジェクトを複数生成
		object_meshes = generate_objects_from_class(class_mesh, num_objects=objects_per_class)

		for o_idx, obj_mesh in enumerate(object_meshes):
			object_dir = os.path.join(class_dir, f"object_{o_idx}")
			models_dir = os.path.join(object_dir, "models")
			os.makedirs(models_dir, exist_ok=True)

			# オブジェクトメッシュにも正規化を適用
			normalize_mesh_AABB(obj_mesh)

			# .obj ファイルとして保存
			save_path = os.path.join(models_dir, "model_normalized.obj")
			# obj_mesh.export(save_path)
			export_obj = trimesh.exchange.obj.export_obj(obj_mesh)
			with open(save_path, 'w') as f:
				f.write(export_obj)
			print(f"Saved: {save_path}")

# -------------------------
# 実行例
# -------------------------
if __name__ == "__main__":
	build_dataset(
		base_dir='DeepSDF/data/synthetic_dataset',
		num_classes=100,        # 任意のクラス数
		objects_per_class=10   # クラスあたりに生成するオブジェクト数
	)
