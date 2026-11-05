"""Shared lightbox and work-surface dimensions, in meters."""
INCH=.0254
WIDTH,DEPTH,HEIGHT=35*INCH,24*INCH,20*INCH
SHEET_SIZE=(18*INCH,12*INCH,.001)
SHEET_CENTER=(0.,DEPTH/2,SHEET_SIZE[2]/2)
WORK_SURFACE_Z=SHEET_SIZE[2]

TRAY_SIZE,TRAY_BASE,TRAY_WALL=.050,.002,.002
TRAY_RIM_HEIGHTS={'reference':.001,'raised':.010}


def tray_geometry(style='reference'):
    """One specification for rendered/colliding trays, planning and metadata."""
    rim=TRAY_RIM_HEIGHTS[style]
    return dict(style=style,color='red',outer_dimensions_m=[TRAY_SIZE]*2,
        base_thickness_m=TRAY_BASE,wall_thickness_m=TRAY_WALL,
        rim_height_above_floor_m=rim,rim_top_above_support_m=TRAY_BASE+rim,
        inside_dimensions_m=[TRAY_SIZE-2*TRAY_WALL]*2,
        # Raised mode lets the cube enter the opening before release, with
        # more than 1 mm jaw-to-rim clearance through the opening motion.
        release_tcp_height_above_mat_m=.032 if style=='raised' else .026)
