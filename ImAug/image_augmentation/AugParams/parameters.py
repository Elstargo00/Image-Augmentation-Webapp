import ast


def get_affine_params(request):

    if not bool(request.POST.get("affine")):
        return None
    
    # Translation degree
    translate_percent = request.POST.get("affine_translate_percent")
    translate_percent = float(ast.literal_eval(translate_percent)) if translate_percent else 0
    # Apply probability
    p = request.POST.get("affine_p")
    p = float(ast.literal_eval(p)) if p else 0.5
    # Rotation degree
    rotate = request.POST.get("affine_rotate")
    rotate = float(ast.literal_eval(rotate)) if rotate else 0
    # Shear degree
    shear = request.POST.get("affine_shear")
    shear = ast.literal_eval(shear) if shear else 0

    return translate_percent, p, rotate, shear


def get_locate_crop_params(request):

    if not bool(request.POST.get("locate_crop")):
        return None
    
    # Croping x min
    x_min = request.POST.get("locate_crop_x_min")
    x_min = int(ast.literal_eval(x_min)) if x_min else 0
    # Croping y min
    y_min = request.POST.get("locate_crop_y_min")
    y_min = int(ast.literal_eval(y_min)) if y_min else 0
    # Croping x max
    x_max = request.POST.get("locate_crop_x_max")
    x_max = int(ast.literal_eval(x_max)) if x_max else 0
    # Croping y min
    y_max = request.POST.get("locate_crop_y_max")
    y_max = int(ast.literal_eval(y_max)) if y_max else 0
    # Apply probability
    p = request.POST.get("locate_crop_p")
    p = float(ast.literal_eval(p)) if p else 0.5

    return x_min, y_min, x_max, y_max, p

def get_random_crop_params(request):

    if not bool(request.POST.get("random_crop")):
        return None
    
    # Croping width
    width = request.POST.get("random_crop_width")
    width = int(ast.literal_eval(width)) if width else 360
    # Croping height
    height = request.POST.get("random_crop_height")
    height = int(ast.literal_eval(height)) if height else 360
    # Apply probability
    p = request.POST.get("random_crop_p")
    p = float(ast.literal_eval(p)) if p else 0.5

    return width, height, p


def get_center_crop_params(request):

    if not bool(request.POST.get("center_crop")):
        return None
    
    # Croping width
    width = request.POST.get("center_crop")
    width = int(ast.literal_eval(width)) if width else 360
    # Croping height
    height = request.POST.get("center_crop")
    height = int(ast.literal_eval(height)) if height else 360
    # Apply probability
    p = request.POST.get("center_crop_p")
    p = float(ast.literal_eval(p)) if p else 0.5

    return width, height, p


def get_horizontal_flip_params(request):

    if not bool(request.POST.get("horizontal_flip")):
        return None
    
    # Apply probability
    p = request.POST.get("horizontal_flip_p")
    p = float(ast.literal_eval(p)) if p else 0.5

    return p


def get_vertical_flip_params(request):

    if not bool(request.POST.get("vertical_flip")):
        return None
    
    # Apply probability
    p = request.POST.get("vertical_flip_p")
    p = float(ast.literal_eval(p)) if p else 0.5

    return p


def get_togray_params(request):

    if not bool(request.POST.get("togray_p")):
        return None
    
    p = request.POST.get("togray_p")
    p = float(ast.literal_eval(p)) if p else 0.5

    return p

def get_gauss_noise_params(request):

    if not bool(request.POST.get("gauss_noise")):
        return None
        
    p = request.POST.get("gauss_noise_p")
    p = float(ast.literal_eval(p)) if p else 0.5
    # Gauss noise mean value
    mean = request.POST.get("gauss_noise_mean")
    mean = float(ast.literal_eval(mean)) if mean else 0
    # Gauss noise variance value
    var_limit = request.POST.get("gauss_noise_var")
    var_limit = float(ast.literal_eval(var_limit)) if var_limit else 0 # NEED FUTURE UPDATE

    return p, mean, var_limit
