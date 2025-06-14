import bpy
import math

# --- パラメータ設定 ---
image_path = "/Users/suzukiakiramuki/playground/被写体.png"  # 画像ファイルのパス
num_tiles = 72  # タイル数
angle_step = math.radians(5)  # 回転角（ラジアン）
scale_step = 1.03  # 拡大率
start_radius = 0.8  # 螺旋の開始半径

# --- シーンをクリーンアップ ---
bpy.ops.object.select_all(action='SELECT')
bpy.ops.object.delete(use_global=False)

# --- 画像テクスチャの作成 ---
img = bpy.data.images.load(image_path)
tex = bpy.data.textures.new("ImageTexture", type='IMAGE')
tex.image = img

# --- マテリアルの作成 ---
mat = bpy.data.materials.new(name="ImageMaterial")
mat.use_nodes = True
bsdf = mat.node_tree.nodes["Principled BSDF"]

# 画像ノード追加
tex_image = mat.node_tree.nodes.new('ShaderNodeTexImage')
tex_image.image = img
mat.node_tree.links.new(bsdf.inputs['Base Color'], tex_image.outputs['Color'])

# --- 基本となる平面の作成 ---
bpy.ops.mesh.primitive_plane_add(size=1)
base_plane = bpy.context.active_object
base_plane.name = "Tile"
base_plane.data.materials.append(mat)

# --- 配列を作成 ---
for i in range(num_tiles):
    angle = i * angle_step
    scale = scale_step ** i
    radius = start_radius * scale
    x = radius * math.cos(angle)
    y = radius * math.sin(angle)

    new_tile = base_plane.copy()
    new_tile.data = base_plane.data.copy()
    new_tile.location = (x, y, 0)
    new_tile.rotation_euler = (0, 0, angle)
    new_tile.scale = (scale, scale, scale)
    bpy.context.collection.objects.link(new_tile)

# --- 元の平面は削除 ---
bpy.data.objects.remove(base_plane, do_unlink=True)