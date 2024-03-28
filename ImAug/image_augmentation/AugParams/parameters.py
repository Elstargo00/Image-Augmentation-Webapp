



def get_affine_params(request):

    if not bool(request.POST.get("affine")):
        return None
    

    translate_percent = request.POST.get("affine_translate_percent")
    translate_percent = float(translate_percent) if translate_percent else 0

    p = request.POST.get("affine_p")
    p = float(p) if p else 0.5

    rotate = 