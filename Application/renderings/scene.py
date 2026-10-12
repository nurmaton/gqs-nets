# Blender scene of one hall: the deployed hall (on its knee walls) and the transport position (in front of the 20-ft container
# door, or on the low-bed semi-trailer), with a 1.80 m person, white background and soft sun shadows on a shadow catcher.
# Writes the render (RGBA, transparent film) and <out>.points.json with the pixel coordinates of named 3-D points and the
# camera (az, el, lens, dist); with --dump <file>.npz every mesh object in world coordinates (after all modifiers) and
# <file>.meta.json for test_scene.py; with --dump-only no render.
# usage: Blender -b --python scene.py -- <geo/hall_*.json> <out.png> [--samples N] [--size WxH] [--az deg] [--el deg] [--lens mm]
#        [--person x,y] [--gap m] [--seed n] [--lines] [--line w] [--view Standard] [--denoise 1] [--adaptive a] [--mask m.png]
#        [--dump d.npz [--dump-only]] [--camera c.json]       the scene data: renderings/make_geo.py
# Written with the assistance of the Claude models Opus 5.5, Fable 5, and Fable 5.1 (Anthropic); all code was checked by the authors.
import bpy, bmesh, json, sys, os, math
from mathutils import Vector, Euler
from bpy_extras.object_utils import world_to_camera_view

argv = sys.argv[sys.argv.index('--') + 1:]
CASE = json.load(open(argv[0])); OUT = argv[1]
def opt(name, default, cast=float):
    return cast(argv[argv.index(name) + 1]) if name in argv else default
SAMPLES = opt('--samples', 128, int); SIZE = opt('--size', '3000x1200', str)
AZ = opt('--az', 38.0); EL = opt('--el', 19.0); LENS = opt('--lens', 55.0)
PERSON = [float(v) for v in opt('--person', '-0.65,0.7', str).split(',')]
GAP = opt('--gap', 3.2); YBIAS = opt('--ybias', 0.0)
PREVIEW = '--preview' in argv
THREE = '--three' in argv and 'mid' in CASE['states']
LINES = '--lines' in argv
LINE_W = opt('--line', 1.0)
GEO = os.path.dirname(os.path.abspath(argv[0]))
W, H = [int(v) for v in SIZE.lower().split('x')]
CONST = json.load(open(os.path.join(GEO, 'constants.json')))
DUMP = opt('--dump', '', str)

# ------------------------------------------------------------------------------------------------ scene reset
bpy.ops.wm.read_factory_settings(use_empty=True)
sc = bpy.context.scene
sc.render.engine = 'CYCLES'
prefs = bpy.context.preferences.addons['cycles'].preferences
prefs.compute_device_type = 'METAL'; prefs.get_devices()
for d in prefs.devices: d.use = (d.type == 'METAL')
sc.cycles.device = 'GPU'; sc.cycles.samples = SAMPLES; sc.cycles.use_denoising = opt('--denoise', 1, int) == 1
try:
    sc.cycles.denoiser = 'OPENIMAGEDENOISE'; sc.cycles.denoising_input_passes = 'RGB_ALBEDO_NORMAL'; sc.cycles.denoising_prefilter = 'ACCURATE'
except Exception as e:
    print('denoiser settings:', e)
sc.cycles.adaptive_threshold = opt('--adaptive', 0.005); sc.cycles.seed = opt('--seed', 0, int)
sc.cycles.use_adaptive_sampling = True
sc.render.resolution_x = W; sc.render.resolution_y = H; sc.render.resolution_percentage = 100
sc.render.film_transparent = True
sc.render.image_settings.file_format = 'PNG'; sc.render.image_settings.color_mode = 'RGBA'
sc.render.filepath = OUT

# ------------------------------------------------------------------------------------------------ helpers
def material(name, rgb, rough=0.55, spec=0.3, alpha=1.0):
    m = bpy.data.materials.new(name); m.use_nodes = True
    b = m.node_tree.nodes['Principled BSDF']
    b.inputs['Base Color'].default_value = (*rgb, 1.0)
    b.inputs['Roughness'].default_value = rough
    for key in ('Specular IOR Level', 'Specular'):
        if key in b.inputs: b.inputs[key].default_value = spec
    if alpha < 1.0:
        b.inputs['Alpha'].default_value = alpha
    return m
for key in ('Coat Weight', 'Clearcoat'):
    pass

MAT_PANEL = material('panel', (0.70, 0.68, 0.635), rough=0.42, spec=0.35)
_b = MAT_PANEL.node_tree.nodes['Principled BSDF']
for _k in ('Coat Weight', 'Clearcoat'):
    if _k in _b.inputs: _b.inputs[_k].default_value = 0.12
MAT_WALL = material('wall', (0.68, 0.68, 0.66), rough=0.85, spec=0.1)
MAT_STEEL = material('steel', (0.20, 0.21, 0.23), rough=0.4, spec=0.5)
MAT_DECK = material('deck', (0.40, 0.41, 0.43), rough=0.65, spec=0.2)
MAT_PERSON = material('person', (0.33, 0.35, 0.40), rough=0.6, spec=0.25)
MAT_RAIL = material('rail', (0.40, 0.41, 0.44), rough=0.4, spec=0.5)
MAT_TYRE = material('tyre', (0.035, 0.035, 0.04), rough=0.85, spec=0.15)
MAT_SKIN = material('skin', (0.60, 0.66, 0.74), rough=0.55, spec=0.3, alpha=0.16)
MAT_ROOF = material('roof', (0.60, 0.66, 0.74), rough=0.55, spec=0.3, alpha=0.09)
MAT_FLOORC = material('cfloor', (0.46, 0.44, 0.40), rough=0.8, spec=0.1)
MAT_WOOD = material('wood', (0.55, 0.42, 0.28), rough=0.75, spec=0.15)
MAT_LIGHT = material('light', (0.62, 0.06, 0.05), rough=0.35, spec=0.4)

def box(name, lo, hi, mat):
    bpy.ops.mesh.primitive_cube_add(size=1.0)
    o = bpy.context.active_object; o.name = name
    o.scale = ((hi[0] - lo[0]), (hi[1] - lo[1]), (hi[2] - lo[2]))
    o.location = ((lo[0] + hi[0]) / 2, (lo[1] + hi[1]) / 2, (lo[2] + hi[2]) / 2)
    o.data.materials.append(mat)
    return o

def cylinder(name, r, depth, loc, rot, mat, verts=48):
    bpy.ops.mesh.primitive_cylinder_add(radius=r, depth=depth, vertices=verts, location=loc, rotation=rot)
    o = bpy.context.active_object; o.name = name; o.data.materials.append(mat)
    bpy.ops.object.shade_smooth()
    return o

def sphere(name, r, loc, mat):
    bpy.ops.mesh.primitive_uv_sphere_add(radius=r, location=loc, segments=48, ring_count=24)
    o = bpy.context.active_object; o.name = name; o.data.materials.append(mat)
    bpy.ops.object.shade_smooth()
    return o

def import_slabs(path, name, mat, loc):
    before = set(bpy.data.objects)
    bpy.ops.wm.obj_import(filepath=path, forward_axis='Y', up_axis='Z')
    new = [o for o in bpy.data.objects if o not in before]
    bpy.ops.object.select_all(action='DESELECT')
    for o in new: o.select_set(True)
    bpy.context.view_layer.objects.active = new[0]
    bpy.ops.object.join()
    o = bpy.context.active_object; o.name = name
    o.data.materials.clear(); o.data.materials.append(mat)
    # exact geometry: the triangulated hulls as written, flat shading, no bevel or smoothing
    for poly in o.data.polygons: poly.use_smooth = False
    o.location = loc
    return o

def person_at(x, y, z=0.0, h=1.80):
    """a plain mannequin of height h standing at (x, y, z), facing -y"""
    s = h / 1.80
    objs = []
    objs.append(sphere('head', 0.105 * s, (x, y, z + 1.695 * s), MAT_PERSON))   # top of the head at z + h
    objs.append(cylinder('neck', 0.045 * s, 0.10 * s, (x, y, z + 1.55 * s), (0, 0, 0), MAT_PERSON))
    t = box('torso', (x - 0.20 * s, y - 0.11 * s, z + 1.03 * s), (x + 0.20 * s, y + 0.11 * s, z + 1.52 * s), MAT_PERSON)
    bev = t.modifiers.new('bevel', 'BEVEL'); bev.width = 0.07 * s; bev.segments = 6; objs.append(t)
    p = box('pelvis', (x - 0.17 * s, y - 0.10 * s, z + 0.86 * s), (x + 0.17 * s, y + 0.10 * s, z + 1.05 * s), MAT_PERSON)
    bev = p.modifiers.new('bevel', 'BEVEL'); bev.width = 0.06 * s; bev.segments = 6; objs.append(p)
    for sgn in (-1, 1):
        objs.append(cylinder('leg', 0.075 * s, 0.90 * s, (x + sgn * 0.095 * s, y, z + 0.47 * s), (0, 0, 0), MAT_PERSON))
        objs.append(box('foot', (x + sgn * 0.095 * s - 0.06 * s, y - 0.16 * s, z), (x + sgn * 0.095 * s + 0.06 * s, y + 0.08 * s, z + 0.06 * s), MAT_PERSON))
        objs.append(cylinder('arm', 0.047 * s, 0.68 * s, (x + sgn * 0.265 * s, y, z + 1.12 * s), (0, math.radians(sgn * 4), 0), MAT_PERSON))
        objs.append(sphere('shoulder', 0.055 * s, (x + sgn * 0.255 * s, y, z + 1.46 * s), MAT_PERSON))
        objs.append(sphere('hand', 0.05 * s, (x + sgn * 0.29 * s, y, z + 0.78 * s), MAT_PERSON))
    bpy.ops.object.select_all(action='DESELECT')
    for o in objs: o.select_set(True)
    bpy.context.view_layer.objects.active = objs[0]
    bpy.ops.object.join(); o = bpy.context.active_object; o.name = 'person'
    return o

# ------------------------------------------------------------------------------------------------ geometry
S = CASE['states']; dep = S['dep']; tr = S['tr']
hw = CASE['hw']; span = CASE['span']; half = span / 2
lift = (-dep['box_lo'][2]) if hw == 0 else 0.0     # m, slab corners below the foot midline
zf = hw + lift                                     # height of the foot midline above the ground
deck = CONST['truck']['vehicle']['well_top_loaded_m'] if CASE['truck'] else CONST['container_floor_top_m']['value']
names = {}          # named 3-D points for the overlay
# deployed hall at the origin, raised by the knee-wall height
o_dep = import_slabs(os.path.join(GEO, dep['obj']), 'deployed', MAT_PANEL, (0, 0, zf))
L_dep = dep['L']
fy = dep['feet_y_ranges']
if hw > 0:
    import numpy as _np
    _Vd = _np.load(os.path.join(GEO, dep['obj'].replace('.obj', '_V.npy')))
    for sgn in (-1, 1):
        wj = CASE['walls'][str(sgn)]                                    # x, height and y-range from the panel material above it
        box('wall%+d' % sgn, (wj['x'][0], wj['y'][0], 0.0), (wj['x'][1], wj['y'][1], wj['height']), MAT_WALL)
# ------------------------------------------------------------------------------------------------ transport props
def build_truck(xc, L):
    """parked 2-axle extendable low-bed (well-type) semi-trailer with the dimensions of the Kaessbohrer K.SLL 2/2H-18/20
    (constants.json truck.vehicle), extended by its full extension and standing on its landing legs, no tractor.
    y: along the trailer; the load (packed net) lies on y = 0 .. L in the middle of the extended well; the rear (axles) is the
    near end (y < 0), the gooseneck the far end.  Sourced: well top (loaded), well length closed + extension, overall length,
    gooseneck length, coupling height, wheelbase, width, axle count, tyre size, well thickness and ground clearance, deck height
    over the axles; tyre diameter, width and twin spacing; axle spacing and width over the tyres from other trailers of the
    same class.  Schematic (not given
    by the maker): king-pin position, gooseneck depth, rear step, suspension hangers, landing legs, underrun bar (UN R58 sizes).
    Parts that touch overlap by 0.4 to 1 mm, so that no two faces coincide; the wooden floor stands 0.5 mm clear of the side
    frames (their tops are level), and the rims sit inside their tyres."""
    T = CONST['truck']; Vh = T['vehicle']; Ty = T['tyre']; Om = T['other_models']; Sc = T['schematic']
    Wt = Vh['width_m']; hw_ = Wt / 2
    top = deck - 1e-4                                            # well floor 0.1 mm below the load
    well = Vh['well_length_closed_m'] + Vh['extension_m']; g = (well - L) / 2
    yw0, yw1 = -g, L + g                                         # extended well
    zb = deck - Vh['well_thickness_m']                           # underside of the well: 0.300 - 0.200 = 0.100 m above the road (maker)
    Gl = Vh['gooseneck_length_m']; H5 = Vh['coupling_height_m']; gd = Sc['gooseneck_depth_m']
    yk = yw1 + Gl - Sc['kingpin_to_front_m']                     # king pin
    wb = Vh['wheelbase_closed_m'] + Vh['extension_m']            # king pin to the centre of the axle group, extended
    sp = Om['axle_spacing_m']; ya = (yk - wb + sp / 2, yk - wb - sp / 2)   # front and rear axle
    yr = (yw1 + Gl) - (Vh['length_closed_m'] + Vh['extension_m'])  # rear end from the maker's overall length (0.85 m behind the rear axle)
    rt = Vh['bogie_deck_height_m']                               # deck over the axles (maker's test article)
    # well (bed): steel structure 0.200 m thick (maker), its top carried by two side frames with a 45 mm wooden floor between them
    ff = 0.045; fw = 0.08                                         # floor thickness (Doll), side-frame width (schematic)
    box('deck', (xc - hw_, yw0, zb), (xc + hw_, yw1, top - ff), MAT_DECK)
    for sgn in (-1, 1):
        box('wellbeam', (xc + sgn * hw_ - (fw if sgn > 0 else 0), yw0, top - ff - 0.0005), (xc + sgn * hw_ + (0 if sgn > 0 else fw), yw1, top), MAT_DECK)
    box('wellfloor', (xc - hw_ + fw + 0.0005, yw0, top - ff - 0.0004), (xc + hw_ - fw - 0.0005, yw1, top), MAT_WOOD)
    # neck up to the gooseneck, gooseneck
    nk = Sc['neck_m']
    box('neck', (xc - hw_ + 0.001, yw1 - 0.001, zb + 0.001), (xc + hw_ - 0.001, yw1 + nk + 0.001, H5 + gd - 0.001), MAT_DECK)
    box('gooseneck', (xc - hw_, yw1 + nk, H5), (xc + hw_, yw1 + Gl, H5 + gd), MAT_DECK)
    cylinder('kingpin', 0.0254, 0.085, (xc, yk, H5 - 0.0425 + 0.0005), (0, 0, 0), MAT_STEEL)
    # rear: step from the well up to the rear deck over the axles, rear deck
    rs = Sc['rear_step_m']
    box('rearstep', (xc - hw_ + 0.001, yw0 - rs, zb + 0.001), (xc + hw_ - 0.001, yw0 + 0.001, rt - 0.001), MAT_DECK)
    box('reardeck', (xc - hw_, yr, rt - 0.06), (xc + hw_, yw0 - rs + 0.001, rt), MAT_DECK)
    for sgn in (-1, 1):                                          # two main beams under the deck over the axles (between the wheels)
        box('rearbeam', (xc + sgn * 0.35 - 0.06, yr + 0.10, rt - 0.24), (xc + sgn * 0.35 + 0.06, yw0 - rs + 0.0005, rt - 0.059), MAT_STEEL)
        box('light', (xc + sgn * 1.075 - 0.125, yr + 0.001, rt - 0.15), (xc + sgn * 1.075 + 0.125, yr + 0.03, rt - 0.059), MAT_LIGHT)
    # running gear: 2 axles, twin tyres 245/70 R17.5, rims 4 mm wider than the tyres, hangers to the rear deck
    R_ = Ty['diameter_m'] / 2; tw = Ty['section_width_m']; ts = Ty['twin_spacing_m']
    xo = Om['width_over_tyres_m'] / 2 - tw / 2; xi = xo - ts      # centres of the outer and inner tyres
    for yy in ya:
        for sgn in (-1, 1):
            for xx in (xo, xi):
                cylinder('tyre', R_, tw, (xc + sgn * xx, yy, R_ + 1e-4), (0, math.radians(90), 0), MAT_TYRE)
                cylinder('rim', 0.222, tw + 0.004, (xc + sgn * xx, yy, R_ + 1e-4), (0, math.radians(90), 0), MAT_STEEL)
            box('hanger', (xc + sgn * 0.55 - 0.05, yy - 0.06, R_ + 1e-4 + 0.049), (xc + sgn * 0.55 + 0.05, yy + 0.06, rt - 0.059), MAT_STEEL)   # 1 mm into the axle and the deck
        cylinder('axle', 0.05, 2 * (xo - (tw + 0.004) / 2) + 0.001, (xc, yy, R_ + 1e-4), (0, math.radians(90), 0), MAT_STEEL)   # through the inner wheels, 0.5 mm into the outer rims
    # landing legs under the gooseneck, 0.06 m in front of the neck, outside the swing clearance circle (schematic)
    yl = yw1 + nk + 0.06
    for sgn in (-1, 1):
        box('leg', (xc + sgn * 0.9 - 0.05, yl - 0.05, 0.049), (xc + sgn * 0.9 + 0.05, yl + 0.05, H5 + 0.001), MAT_STEEL)
        box('leg_pad', (xc + sgn * 0.9 - 0.15, yl - 0.15, 1e-4), (xc + sgn * 0.9 + 0.15, yl + 0.15, 0.05), MAT_STEEL)
    # mud flaps 0.10 m behind the rear twin wheels
    for sgn in (-1, 1):
        box('mudflap', (xc + sgn * (xo + xi) / 2 - 0.26, ya[1] - R_ - 0.11, 0.25), (xc + sgn * (xo + xi) / 2 + 0.26, ya[1] - R_ - 0.10, rt - 0.059), MAT_TYRE)
    # rear underrun bar (UN R58 sizes) with two struts to the rear deck
    # (behind the rear tyres, whose rear edge is 0.39 m in front of the rear end; struts inside the inner tyres)
    box('underrun', (xc - 1.17, yr + 0.005, 0.33), (xc + 1.17, yr + 0.095, 0.45), MAT_STEEL)
    for sgn in (-1, 1):
        box('underrun_strut', (xc + sgn * 0.5 - 0.04, yr + 0.015, 0.449), (xc + sgn * 0.5 + 0.04, yr + 0.085, rt - 0.059), MAT_STEEL)
    return dict(w=Wt, y_rear=yr, y_front=yw1 + Gl, yk=yk, axles=ya, well=(yw0, yw1), rear_step=rs)


def build_container(xc, L=None):
    """a 20-ft container: ISO 668 1CC minimum internal size and door opening, nominal external size; translucent walls and
    roof, open end with the door frame, a schematic underframe (bottom side rails, cross-members, floor plate), corner posts,
    rails and castings.  The corner castings carry the nominal external size exactly (as in a real container); posts and rails
    sit 0.5 mm and the skins 1 mm inside it, so that no two faces coincide; the translucent roof, side walls and rear wall
    are separated by 0.5 mm gaps, and the rear wall stands 0.5 mm above the floor plate.  Floor top 0.1 mm below the packed net."""
    C = CONST['container_20ft']; (CL, CW, CH), (IL, IW, IH), (DW, DH) = C['external_L_W_H_m'], C['internal_min_L_W_H_m'], C['door_min_W_H_m']
    wt = (CW - IW) / 2; ft = deck - 1e-4
    y0 = -(CL - IL) / 2
    lo = (xc - CW / 2, y0, 0.0); hi = (xc + CW / 2, y0 + CL, CH)
    # floor plate and underframe
    box('c_floorplate', (lo[0] + wt, lo[1] + 0.012, ft - 0.03), (hi[0] - wt, hi[1] - 0.012, ft), MAT_FLOORC)
    for sgn in (-1, 1):
        xa_, xb_ = sorted((xc + sgn * (CW / 2 - 0.002), xc + sgn * (CW / 2 - 0.16)))
        box('c_siderail', (xa_, lo[1] + 0.002, 0.0), (xb_, hi[1] - 0.002, ft - 0.0302), MAT_STEEL)
    nx = 14
    for k in range(nx):
        yy = lo[1] + 0.3 + k * (CL - 0.6) / (nx - 1)
        box('c_cross', (lo[0] + 0.161, yy - 0.04, ft - 0.13), (hi[0] - 0.161, yy + 0.04, ft - 0.0302), MAT_STEEL)
    for yy, sgn in ((lo[1] + 0.002, 1), (hi[1] - 0.002, -1)):
        ya_, yb_ = sorted((yy, yy + sgn * 0.15))
        box('c_endrail', (lo[0] + 0.161, ya_, 0.0), (hi[0] - 0.161, yb_, ft - 0.0302), MAT_STEEL)
    # skins
    sk = 0.001                                                    # skins 1 mm inside the nominal external size
    box('c_wall_l', (lo[0] + sk, lo[1] + sk, ft), (lo[0] + wt, hi[1] - sk, CH - sk), MAT_SKIN)
    box('c_wall_r', (hi[0] - wt, lo[1] + sk, ft), (hi[0] - sk, hi[1] - sk, CH - sk), MAT_SKIN)
    g = 0.0005                                                    # translucent parts never share a face: 0.5 mm gaps (below a pixel)
    box('c_wall_rear', (lo[0] + wt + g, IL, ft + g), (hi[0] - wt - g, hi[1] - sk, CH - sk), MAT_SKIN)
    box('c_roof', (lo[0] + wt + g, lo[1] + sk, ft + IH), (hi[0] - wt - g, IL - g, CH - sk), MAT_ROOF)
    # door frame at the open end: posts and header leaving the minimum door opening DW x DH above the floor
    fd = -y0                                                      # the door frame ends where the ISO internal length starts (y = 0)
    for sgn in (-1, 1):
        xa_, xb_ = sorted((xc + sgn * DW / 2, xc + sgn * (IW / 2 - 0.001)))
        box('c_doorpost', (xa_, y0 + 0.002, ft), (xb_, y0 + fd, ft + IH - 0.001), MAT_STEEL)
    box('c_header', (xc - IW / 2 + 0.001, y0 + 0.002, ft + DH), (xc + IW / 2 - 0.001, y0 + fd, ft + IH - 0.001), MAT_STEEL)
    # corner posts and top rails 0.5 mm inside the nominal external size (p < 0), bottom on the ground; the castings
    # (0.5 mm larger on every outer side) end exactly on the nominal external size
    r = 0.04; p = -0.0005
    X = (lo[0] - p, hi[0] + p); Y = (lo[1] - p, hi[1] + p)
    for x in X:
        for y in Y:
            # corner posts: inner face 1 mm outside the door opening (the clear opening stays DW wide)
            xin = (xc - DW / 2 - 0.001) if x < xc else (xc + DW / 2 + 0.001)
            box('c_post', (min(x, xin), min(y, y + (2 * r if y < 0 else -2 * r)), 0.0),
                (max(x, xin), max(y, y + (2 * r if y < 0 else -2 * r)), CH + p), MAT_STEEL)
    for y in Y:
        y2 = y + (2 * r if y < 0 else -2 * r)
        box('c_toprail_x', (X[0] + 0.002, min(y, y2), CH - 2 * r), (X[1] - 0.002, max(y, y2), CH + p), MAT_STEEL)
    for x in X:
        x2 = x + (2 * r if x < xc else -2 * r)
        box('c_toprail_y', (min(x, x2), Y[0] + 0.002, CH - 2 * r), (max(x, x2), Y[1] - 0.002, CH + p), MAT_STEEL)
    for x in X:
        for y in Y:
            box('c_cast', (min(x, x + (0.162 if x < xc else -0.162)) - 0.0005, min(y, y + (0.178 if y < 0 else -0.178)) - 0.0005, 0.0),
                (max(x, x + (0.162 if x < xc else -0.162)) + 0.0005, max(y, y + (0.178 if y < 0 else -0.178)) + 0.0005, 0.118), MAT_STEEL)
            box('c_cast', (min(x, x + (0.162 if x < xc else -0.162)) - 0.0005, min(y, y + (0.178 if y < 0 else -0.178)) - 0.0005, CH - 0.118),
                (max(x, x + (0.162 if x < xc else -0.162)) + 0.0005, max(y, y + (0.178 if y < 0 else -0.178)) + 0.0005, CH + p + 0.0005), MAT_STEEL)
    return dict(w=CW, door=dict(tl=(xc - DW / 2, y0, ft + DH), tr=(xc + DW / 2, y0, ft + DH), bl=(xc - DW / 2, y0, ft), br=(xc + DW / 2, y0, ft)),
                cont=dict(tl=(lo[0], lo[1], CH), tr=(hi[0], lo[1], CH), tl_far=(lo[0], hi[1], CH)))


def tube_edges(name, lo, hi, r, mat):
    X = (lo[0], hi[0]); Y = (lo[1], hi[1]); Z = (lo[2], hi[2])
    for y in Y:
        for z in Z: box(name, (lo[0] - r, y - r, z - r), (hi[0] + r, y + r, z + r), mat)
    for x in X:
        for z in Z: box(name, (x - r, lo[1] - r, z - r), (x + r, hi[1] + r, z + r), mat)
    for x in X:
        for y in Y: box(name, (x - r, y - r, lo[2] - r), (x + r, y + r, hi[2] + r), mat)


FOUR = '--four' in argv
L_tr = tr['L']
cw_prop = 2.55 if CASE['truck'] else 2.438
if FOUR:
    # left to right: the packed net alone on the ground, the empty container (or truck), the loaded one, the hall
    x_pack = -(half + GAP + cw_prop / 2)                      # loaded container / truck
    x_empty = x_pack - cw_prop / 2 - GAP - cw_prop / 2
    x_alone = x_empty - cw_prop / 2 - GAP - tr['W'] / 2
    import_slabs(os.path.join(GEO, tr['obj']), 'packed_alone', MAT_PANEL, (x_alone, 0, 0.0))
    (build_truck if CASE['truck'] else build_container)(x_empty, L_tr)
elif THREE:
    mid = S['mid']; gap2 = GAP * 0.6
    x_mid = -(half + gap2 + mid['W'] / 2)
    x_pack = x_mid - mid['W'] / 2 - gap2 - cw_prop / 2
    lift_mid = -mid['box_lo'][2]
    import_slabs(os.path.join(GEO, mid['obj']), 'intermediate', MAT_PANEL, (x_mid, 0, lift_mid))
else:
    x_pack = -(half + GAP + tr['W'] / 2)
o_tr = import_slabs(os.path.join(GEO, tr['obj']), 'packed', MAT_PANEL, (x_pack, 0, deck))
if CASE['truck']:
    TRK = build_truck(x_pack, L_tr)
else:
    info = build_container(x_pack)
    for k_, v_ in info['door'].items(): names['door_' + k_] = v_
    for k_, v_ in info['cont'].items(): names['cont_' + k_] = v_
# person in the near opening of the deployed hall
person_at(PERSON[0], PERSON[1], 0.0, h=CONST['person_height_m']['value'])
# floor (shadow catcher)
bpy.ops.mesh.primitive_plane_add(size=400, location=(0, 0, -1e-4))   # ground 0.1 mm below z = 0 (no coincident faces)
floor = bpy.context.active_object; floor.name = 'floor'; floor.is_shadow_catcher = True
floor.data.materials.append(material('floor', (1, 1, 1), rough=1.0, spec=0.0))

# ------------------------------------------------------------------------------------------------ named points
yf = -0.45                                            # dimension lines a little in front of the open ends
import numpy as np
Vd = np.load(os.path.join(GEO, dep['obj'].replace('.obj', '_V.npy'))); Vt = np.load(os.path.join(GEO, tr['obj'].replace('.obj', '_V.npy')))
mid_x = np.array(dep['mid_x']); mid_z = np.array(dep['mid_z']); ja = int(np.argmax(mid_z))
y_crown = float(Vd[:, ja, 1].min())                      # front end of the crown vertex row
y_f1 = float(Vd[:, 0, 1].min()); y_f2 = float(Vd[:, -1, 1].min())
fR, yR = (dep['foot2'], y_f2) if dep['foot2'][0] > dep['foot1'][0] else (dep['foot1'], y_f1)   # the foot on the right (+x)
names.update({
    'span_l': (dep['foot1'][0], min(y_f1, y_f2) + yf, 0.0), 'span_r': (dep['foot2'][0], min(y_f1, y_f2) + yf, 0.0),
    'foot1_line': (dep['foot1'][0], y_f1, zf), 'foot2_line': (dep['foot2'][0], y_f2, zf),
    'foot1_ground': (dep['foot1'][0], y_f1, 0.0), 'foot2_ground': (dep['foot2'][0], y_f2, 0.0),
    'rise_b': (fR[0] + 1.15, yR + yf, zf), 'rise_t': (fR[0] + 1.15, yR + yf, zf + CASE['rise']),
    'rise_b_ext': (fR[0], yR, zf),
    'wall_b': (fR[0] + 0.55, yR + yf, 0.0), 'wall_t': (fR[0] + 0.55, yR + yf, hw),
    'apex': (float(mid_x[ja]), y_crown, zf + float(mid_z[ja])),
    'len_a': (float(mid_x[ja]), 0.0, zf + float(mid_z[ja]) + 0.55), 'len_b': (float(mid_x[ja]), L_dep, zf + float(mid_z[ja]) + 0.55),
    'origin': (0, 0, 0), 'person_top': (PERSON[0], PERSON[1], 1.80), 'person_feet': (PERSON[0], PERSON[1], 0.0),
})
# packed net: extreme vertices of the transport mesh (front-most among the extreme ones), in world coordinates
x_dim, z_dim = (x_alone, 0.0) if FOUR else (x_pack, deck)        # the packed net that carries W and H
Pt = Vt.reshape(-1, 3) + np.array([x_dim, 0.0, z_dim])
def front_extreme(axis, sign):
    v = sign * Pt[:, axis]; c = Pt[v >= v.max() - 1e-3]; return c[np.argmin(c[:, 1])]
pl, pr, pt_ = front_extreme(0, -1), front_extreme(0, 1), front_extreme(2, 1)
xr_arrow = x_dim + tr['box_hi'][0] + (0.95 if not CASE['truck'] else 0.6)
names.update({
    'pack_l': (x_dim + tr['box_lo'][0], -0.25, z_dim), 'pack_r': (x_dim + tr['box_hi'][0], -0.25, z_dim),
    'pack_l_ext': tuple(pl), 'pack_r_ext': tuple(pr),
    'pack_b': (xr_arrow, -0.25, z_dim), 'pack_t': (xr_arrow, -0.25, z_dim + tr['H']),
    'pack_t_ext': tuple(pt_), 'pack_b_ext': (x_dim + tr['box_hi'][0], -0.25, z_dim),
    'carrier_text': (x_pack, -1.0, 0.0),
})
if CASE['truck']:
    # on the low-bed the near end of the load is behind the rear section: anchors on the load's near face (y = 0), the
    # well arrow beside the trailer, the carrier text in front of the trailer's rear end
    # the loaded trailer's dimensions as one chain on the side of the well that faces the camera (x < 0 side), 3.0 m along the
    # load, where the well is not hidden by the rear section: road -> well underside -> well top (= base of the load) -> top of the load
    dw = CONST['truck']['vehicle']['width_m']; clr = CONST['truck']['vehicle']['ground_clearance_m']
    Yd = 3.0; xl = x_pack - dw / 2; xc_ = xl - 0.5
    names.update({'chain_0': (xc_, Yd, 0.0), 'chain_1': (xc_, Yd, clr), 'chain_2': (xc_, Yd, deck), 'chain_3': (xc_, Yd, deck + tr['H']),
                  'well_under': (xl, Yd, clr), 'well_top': (xl, Yd, deck)})
    names['carrier_text'] = (x_pack, TRK['y_rear'] - 0.8, 0.0)
# ------------------------------------------------------------------------------------------------ light + world
wd = bpy.data.worlds.new('w'); sc.world = wd; wd.use_nodes = True
bg = wd.node_tree.nodes['Background']; bg.inputs[0].default_value = (1, 1, 1, 1); bg.inputs[1].default_value = 0.8
bpy.ops.object.light_add(type='SUN', location=(0, 0, 10))
sun = bpy.context.active_object; sun.data.energy = 2.3; sun.data.angle = math.radians(11.0)
sun.rotation_euler = Euler((math.radians(48), 0, math.radians(-128)), 'XYZ')   # from the left and slightly behind, 42 deg high
# a large soft fill from the front-right, so the shaded faces keep some form
bpy.ops.object.light_add(type='AREA', location=(14, -14, 10))
fill = bpy.context.active_object; fill.data.energy = 2500; fill.data.size = 12
fill.rotation_euler = (Vector((0, 0, 0)) - fill.location).to_track_quat('-Z', 'Y').to_euler()
sc.view_settings.view_transform = opt('--view', 'AgX', str); sc.view_settings.exposure = opt('--exposure', 0.0)

# ------------------------------------------------------------------------------------------------ camera
bpy.ops.object.camera_add(); cam = bpy.context.active_object; sc.camera = cam
cam.data.lens = LENS; cam.data.sensor_width = 36; cam.data.clip_end = 1000
objs_fit = [o for o in bpy.data.objects if o.type == 'MESH' and o.name != 'floor']
def corners():
    pts = []
    for o in objs_fit:
        for c in o.bound_box: pts.append(o.matrix_world @ Vector(c))
    return pts
pts = corners()
# the dimension points, with room for their labels (above the ridge arrows, below the packed-width arrow)
for k, p in names.items():
    pts.append(Vector(p))
for k in ('len_a', 'len_b'):
    pts.append(Vector(names[k]) + Vector((0, 0, 0.55)))
for k in ('pack_l', 'pack_r'):
    pts.append(Vector(names[k]) + Vector((0, -1.6, -1.6)))       # room for the two-line text under the packed net
# room for the labels to the right of the rise and wall arrows and of the panel-thickness leader
for k in [k_ for k_ in ('rise_b', 'rise_t', 'wall_b', 'wall_t', 'pack_b', 'pack_t', 'deck_b', 'deck_t') if k_ in names]:
    pts.append(Vector(names[k]) + Vector((3.0, 0, 0)))
# the truck's dimension chain has its labels on the left
for k in [k_ for k_ in ('chain_0', 'chain_2', 'chain_3') if k_ in names and not (FOUR and k_ == 'chain_3')]:
    pts.append(Vector(names[k]) + Vector((-2.0, 0, 0)))
cen = sum(pts, Vector()) / len(pts)
d = Vector((-math.sin(math.radians(AZ)) * math.cos(math.radians(EL)), -math.cos(math.radians(AZ)) * math.cos(math.radians(EL)), math.sin(math.radians(EL))))
dist = 30.0
margin = 0.03
for it in range(6):
    cam.location = cen + d * dist
    cam.rotation_euler = (cen - cam.location).to_track_quat('-Z', 'Y').to_euler()
    bpy.context.view_layer.update()
    uv = [world_to_camera_view(sc, cam, p) for p in pts]
    xs = [p.x for p in uv]; ys = [p.y for p in uv]
    wx, wy = max(xs) - min(xs), max(ys) - min(ys)
    f = max(wx / (1 - 2 * margin), wy / (1 - 2 * margin))
    # recentre: shift the target in the camera plane
    cx, cy = (max(xs) + min(xs)) / 2 - 0.5, (max(ys) + min(ys)) / 2 - 0.5 - YBIAS
    right = cam.matrix_world.to_quaternion() @ Vector((1, 0, 0)); up = cam.matrix_world.to_quaternion() @ Vector((0, 1, 0))
    # pixel-to-world scale at the target distance
    fw = dist * cam.data.sensor_width / cam.data.lens; fh = fw * H / W
    cen = cen + right * (cx * fw) + up * (cy * fh)
    dist = dist * f
print('camera dist %.2f, centre %s, deployed lift %.3f m' % (dist, [round(v, 2) for v in cen], lift))
CAMFILE = opt('--camera', '', str)
if CAMFILE:                                                       # the camera as set up here (written before any rendering)
    json.dump(dict(case=CASE['name'], size=[W, H], az=AZ, el=EL, lens=cam.data.lens, sensor_width=cam.data.sensor_width,
                   sensor_fit=cam.data.sensor_fit, dist=dist, target=list(cen), location=list(cam.location),
                   rotation_euler_xyz=list(cam.rotation_euler), clip_end=cam.data.clip_end), open(CAMFILE, 'w'), indent=1)

# ------------------------------------------------------------------------------------------------ visibility helper
SKIN_NAMES = ('c_wall', 'c_roof', 'envelope')
def visible(p, tol=0.01):
    """True if the 3-D point p is seen from the camera, looking through translucent skins"""
    dg2 = bpy.context.evaluated_depsgraph_get(); org = cam.location.copy(); tgt = Vector(p)
    for _ in range(8):
        dv = tgt - org; L_ = dv.length; dv.normalize()
        hit, loc, nrm, idx, ob, mtx = sc.ray_cast(dg2, org, dv, distance=L_ + 0.05)
        if not hit: return True
        if (loc - tgt).length < tol: return True
        if ob.name.startswith(SKIN_NAMES): org = loc + dv * 1e-4; continue
        return False
    return False
# ------------------------------------------------------------------------------------------------ 80 mm anchor (ray cast)
bpy.context.view_layer.update()
dg_ = bpy.context.evaluated_depsgraph_get()
Sd = np.load(os.path.join(GEO, '%s_dep_slabs_verified.npy' % CASE['name']), allow_pickle=True)
cands = []
i0, i1 = (0, 1) if Vd[0, :, 1].mean() < Vd[-1, :, 1].mean() else (-1, -2)     # the vertex column at the near (camera) end
for j in range(Vd.shape[1] - 1):
    a3, b3 = Vd[i0, j] + np.array([0, 0, zf]), Vd[i0, j + 1] + np.array([0, 0, zf])
    m3 = (a3 + b3) / 2
    q = np.array([Vd[i0, j], Vd[i1, j], Vd[i1, j + 1], Vd[i0, j + 1]]) + np.array([0, 0, zf])
    nf = np.cross(q[2] - q[0], q[3] - q[1]); nf /= np.linalg.norm(nf)
    # inward = towards the slab body (its points lie on one side of the face plane)
    ki = int(np.argmin(np.linalg.norm(np.array([np.asarray(s_, float).mean(0) for s_ in Sd]) + np.array([0, 0, zf]) - q.mean(0), axis=1)))
    if (np.asarray(Sd[ki], float).mean(0) + np.array([0, 0, zf]) - q.mean(0)) @ nf < 0: nf = -nf
    c3 = m3 + 0.04 * nf
    u = ((q[1] + q[2]) / 2 - m3); u -= (u @ nf) * nf; side_n = -u / np.linalg.norm(u)
    cands.append((c3[2], c3, side_n))
anchor = None
for z_, c3, side_n in sorted(cands, key=lambda t: -t[0]):
    org = cam.location; dvec = Vector(c3) - org; dist_ = dvec.length; dvec.normalize()
    hit, loc, nrm, idx, ob, mtx = sc.ray_cast(dg_, org, dvec, distance=dist_ + 0.05)
    if hit and ob.name == 'deployed' and (loc - Vector(c3)).length < 0.02 and nrm.dot(Vector(side_n)) > 0.9 and dvec.dot(Vector(side_n)) < -0.2:
        anchor = c3; break
names['thick_anchor'] = tuple(anchor) if anchor is not None else tuple(cands[0][1])
names['apex'] = tuple((Vd[i0, ja] + Vd[i1, ja]) / 2 + np.array([0, 0, zf]))
print('apex anchor %s visible: %s' % (np.round(names['apex'], 3), visible(names['apex'])))
# foot lines: midpoint of the first boundary edge of each foot row at the near end (it lies on the foot line and on the slab edge)
for nm_, j_ in (('foot_a', 0), ('foot_b', -1)):
    pm = (Vd[i0, j_] + Vd[i1, j_]) / 2 + np.array([0, 0, zf]); names[nm_] = tuple(pm); names[nm_ + '_ground'] = (pm[0], pm[1], 0.0)
fa, fb = np.array(names['foot_a']), np.array(names['foot_b'])
fl, fr = (fa, fb) if fa[0] < fb[0] else (fb, fa)
names['foot1_line'] = tuple(fl); names['foot2_line'] = tuple(fr)
names['rise_b_ext'] = tuple(fr)
yS = min(0.0, fl[1], fr[1]) + yf
names['span_l'] = (fl[0], yS, 0.0); names['span_r'] = (fr[0], yS, 0.0)
names['foot1_ground'] = (fl[0], fl[1], 0.0); names['foot2_ground'] = (fr[0], fr[1], 0.0)
xr_ = fr[0] + 1.15; yr_ = fr[1]          # R and the feet chain in the plane of the foot anchor (clear of S)
if hw > 0:
    yr_ = CASE['walls']['1' if fr[0] > 0 else '-1']['y'][0]          # the plane of the wall's front face
names['rise_b'] = (xr_, yr_, zf); names['rise_t'] = (xr_, yr_, zf + CASE['rise'])
names['wall_b'] = (xr_, yr_, 0.0); names['wall_t'] = (xr_, yr_, zf)          # chain dimension under R: ground -> foot line
if hw > 0:
    wR = CASE['walls']['1' if fr[0] > 0 else '-1']
    names['wall_corner'] = (wR['x'][1] if fr[0] > 0 else wR['x'][0], wR['y'][0], 0.0)
# overall L: extension lines from the extreme panel points (in the planes y = 0 and y = L) to the arrow ends
Sd_all = np.concatenate([np.asarray(s_, float) for s_ in Sd]) + np.array([0, 0, zf])
for nm_, sel in (('len_a_ext', Sd_all[:, 1] <= Sd_all[:, 1].min() + 1e-6), ('len_b_ext', Sd_all[:, 1] >= Sd_all[:, 1].max() - 1e-6)):
    cand = Sd_all[sel]; tgt_ = np.array(names['len_a' if nm_ == 'len_a_ext' else 'len_b'])
    names[nm_] = tuple(cand[np.argmin(np.linalg.norm(cand[:, [0, 2]] - tgt_[[0, 2]], axis=1))])
# packed net: H top on a visible top vertex; W arrow clear of the door frame
St_all = np.concatenate([np.asarray(s_, float) for s_ in np.load(os.path.join(GEO, '%s_tr_slabs_verified.npy' % CASE['name']), allow_pickle=True)]) + np.array([x_dim, 0, z_dim])
ztop = St_all[:, 2].max(); tops = St_all[St_all[:, 2] >= ztop - 1e-4]; tops = tops[np.argsort(tops[:, 1])]
vis_top = next((t_ for t_ in tops if visible(t_)), tops[0])
names['pack_t_ext'] = tuple(vis_top); print('H top anchor %s visible: %s' % (np.round(vis_top, 3), visible(vis_top)))
if CASE['truck'] and not FOUR:
    tops_d = tops[np.argsort(np.abs(tops[:, 1] - names['chain_3'][1]))]
    lt_ = next((t_ for t_ in tops_d if visible(t_)), tops_d[0]); names['load_top'] = tuple(lt_)
    for k_ in ('chain_0', 'chain_1', 'chain_2', 'well_under', 'well_top', 'load_top'):
        print('truck anchor %s %s visible: %s' % (k_, np.round(names[k_], 3), visible(names[k_]) if k_ in ('well_under', 'well_top', 'load_top') else 'free space'))
yW = -0.45 if FOUR else (0.0 if CASE['truck'] else -0.85)
zW = z_dim if (FOUR or not CASE['truck']) else CONST['truck']['vehicle']['bogie_deck_height_m'] + 0.10   # above the trailer's rear section
names['pack_l'] = (x_dim + tr['box_lo'][0], yW, zW); names['pack_r'] = (x_dim + tr['box_hi'][0], yW, zW)
print('80 mm anchor %s on a visible side face: %s' % (np.round(names['thick_anchor'], 3), anchor is not None))
# ------------------------------------------------------------------------------------------------ outlines (optional)
if LINES:
    # hinge lines as line art on the true panel edges (Freestyle on the panel objects only); no geometry is changed
    coll = bpy.data.collections.new('panels'); sc.collection.children.link(coll)
    for o in bpy.data.objects:
        if o.name in ('deployed', 'packed', 'packed_alone'): coll.objects.link(o)
    sc.render.use_freestyle = True; sc.render.line_thickness_mode = 'ABSOLUTE'; sc.render.line_thickness = LINE_W
    vl = bpy.context.view_layer; vl.use_freestyle = True
    fs = vl.freestyle_settings; fs.crease_angle = math.radians(150)
    for ls_ in list(fs.linesets): fs.linesets.remove(ls_)
    ls = fs.linesets.new('panel_edges')
    ls.select_by_collection = True; ls.collection = coll
    ls.select_silhouette = True; ls.select_border = True; ls.select_crease = True
    ls.linestyle.color = (0.32, 0.33, 0.36); ls.linestyle.thickness = LINE_W; ls.linestyle.alpha = 0.75
# ------------------------------------------------------------------------------------------------ render
if PREVIEW:
    sc.cycles.samples = 24; sc.render.resolution_percentage = 50
pct = sc.render.resolution_percentage / 100.0
for o in (o_dep, o_tr):
    bb = [o.matrix_world @ Vector(c) for c in o.bound_box]
    lo = [min(v[i] for v in bb) for i in range(3)]; hi = [max(v[i] for v in bb) for i in range(3)]
    print('bbox %-9s lo %s hi %s' % (o.name, [round(v, 3) for v in lo], [round(v, 3) for v in hi]))
bpy.context.view_layer.update()
placements = {'deployed': [0.0, 0.0, zf], 'packed': [x_pack, 0.0, deck]}
if FOUR: placements['packed_alone'] = [x_alone, 0.0, 0.0]
if DUMP:
    import numpy as np
    dg = bpy.context.evaluated_depsgraph_get(); arr = {}
    for o in bpy.data.objects:
        if o.type != 'MESH': continue
        oe = o.evaluated_get(dg); me = oe.to_mesh(); me.calc_loop_triangles()
        Mw = oe.matrix_world
        arr[o.name + '__V'] = np.array([tuple(Mw @ v.co) for v in me.vertices], float)
        arr[o.name + '__F'] = np.array([tuple(t.vertices) for t in me.loop_triangles], int)
        oe.to_mesh_clear()
    np.savez_compressed(DUMP, **arr)
    json.dump(dict(placements=placements, case=CASE['name'], four=FOUR, deck=deck, zf=zf, hw=hw, lift=lift, truck=CASE['truck']),
              open(os.path.splitext(DUMP)[0] + '.meta.json', 'w'), indent=1)
    print('dumped', DUMP)
    if '--dump-only' in argv: sys.exit(0)
bpy.ops.render.render(write_still=True)
W, H = int(round(W * pct)), int(round(H * pct))
MASK = opt('--mask', '', str)
if MASK:
    eng = sc.render.engine; fsx = sc.render.use_freestyle
    sc.render.engine = 'BLENDER_WORKBENCH'; sc.render.use_freestyle = False
    sh = sc.display.shading; sh.light = 'FLAT'; sh.color_type = 'SINGLE'; sh.single_color = (1, 1, 1)
    sc.render.film_transparent = True; floor.hide_render = True; vt = sc.view_settings.view_transform; sc.view_settings.view_transform = 'Standard'
    out0 = sc.render.filepath; sc.render.filepath = MASK; bpy.ops.render.render(write_still=True)
    hidden0 = {o.name: o.hide_render for o in bpy.data.objects}
    extra_mask = os.environ.get('MASK_PREFIX', '')
    tags = [('person', lambda n: n == 'person'), ('panels', lambda n: n in ('deployed', 'packed', 'packed_alone'))]
    if extra_mask: tags.append((extra_mask, lambda n, p=extra_mask: n.split('.')[0] == p))
    for tag_, keep in tags:
        for o in bpy.data.objects:
            if o.type == 'MESH': o.hide_render = not keep(o.name)
        sc.render.filepath = os.path.splitext(MASK)[0] + '_%s.png' % tag_; bpy.ops.render.render(write_still=True)
    for o in bpy.data.objects: o.hide_render = hidden0[o.name]
    sc.render.filepath = out0; sc.render.engine = eng; sc.render.use_freestyle = fsx; floor.hide_render = False; sc.view_settings.view_transform = vt
    print('mask', MASK)
pix = {}
for k, p in names.items():
    v = world_to_camera_view(sc, cam, Vector(p))
    pix[k] = [v.x * W, (1 - v.y) * H]
# every label value with its source
th80 = CASE.get('panel_thickness_m', 0.08)
labels = dict(
    span=dict(value=span, source='geo/%s.json span: foot-line midpoints of the verified deployed positions' % CASE['name']),
    rise=dict(value=CASE['rise'], source='geo/%s.json rise: midline apex above the foot line, verified deployed positions' % CASE['name']),
    L_dep=dict(value=L_dep, source='geo/%s.json states.dep.L: extent along the hall of the rendered panels' % CASE['name']),
    L_full=dict(value=float(Vd[:, :, 1].max(0).min() - Vd[:, :, 1].min(0).max()), source='overlap of all vertex rows along the hall (geo/%s_dep_V.npy); hall length per container' % CASE['name']),
    W_tr=dict(value=tr['W'], source='geo/%s.json states.tr.W: width of the verified transport state with its 80 mm panels' % CASE['name']),
    H_tr=dict(value=tr['H'], source='geo/%s.json states.tr.H: height of the verified transport state with its 80 mm panels' % CASE['name']),
    L_tr=dict(value=L_tr, source='geo/%s.json states.tr.L: extent along the hall of the transport state' % CASE['name']),
    thickness=dict(value=th80, source='measured thickness of the verified slabs (outer to inner skin), geometry test'),
    footline=dict(value=hw, source=CONST['knee_walls_footline_m']['source']) if hw > 0 else None,
    container=dict(value=CONST['container_20ft']['door_min_W_H_m'], source=CONST['container_20ft']['source']) if not CASE['truck'] else None,
    truck_width=dict(value=CONST['truck']['vehicle']['width_m'], source=CONST['truck']['vehicle']['name'] + ': ' + CONST['truck']['vehicle']['source']) if CASE['truck'] else None,
    road_height=dict(value=deck + tr['H'], limit=CONST['truck']['max_height_m'], source='well top (' + CONST['truck']['vehicle']['source'] + ') + H_tr; limit: ' + CONST['truck']['source']) if CASE['truck'] else None,
    deck=dict(value=deck, source=CONST['truck']['vehicle']['name'] + ', well height loaded: ' + CONST['truck']['vehicle']['source']) if CASE['truck'] else None,
    clearance=dict(value=CONST['truck']['vehicle']['ground_clearance_m'], source=CONST['truck']['vehicle']['ground_clearance_source']) if CASE['truck'] else None,
    well_thickness=dict(value=CONST['truck']['vehicle']['well_thickness_m'], source=CONST['truck']['vehicle']['well_thickness_source']) if CASE['truck'] else None,
)
json.dump(dict(size=[W, H], points=pix, cam=dict(az=AZ, el=EL, lens=LENS, dist=dist), x_pack=x_pack, deck=deck, hw=hw, case=CASE['name'],
               W_tr=tr['W'], H_tr=tr['H'], L_tr=L_tr, L_dep=L_dep, span=span, rise=CASE['rise'], labels=labels, placements=placements),
          open(os.path.splitext(OUT)[0] + '.points.json', 'w'), indent=1)
print('wrote', OUT)
