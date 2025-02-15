INPUT_SCHEMA = {
    'seed': {
        'type': int,
        'required': False,
        'default': None
    },
    'image': {
    	'type': str,
        'required': True,
        'default': None
    },
    'simplify': {
        'type': float,
        'required': True,
        'default': None
    },
    'texture_size': {
        'type': int,
        'required': True,
        'default': None
    }
}
