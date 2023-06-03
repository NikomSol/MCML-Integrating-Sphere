import json

import jsonschema
import pytest


@pytest.fixture()
def cfg():
    return json.load(open('example.json'))


@pytest.fixture()
def schema():
    return json.load(open('schema.json'))


@pytest.fixture()
def types_and_values():
    return [
        ('string', '1'),
        ('number', 1.0),
        ('integer', 1),
        ('object', dict()),
        ('array', []),
        ('boolean', True),
        ('boolean', False),
    ]


def test_cfg_is_valid(cfg, schema):
    jsonschema.validate(instance=cfg, schema=schema)


EXCESSIVE_KEY = 'kwakwakwa'
MIN_KEY_STR = 'minimum'
PROPERTIES_KEY_STR = 'properties'
REF_KEY_STR = '$ref'
TYPE_KEY_STR = 'type'


@pytest.mark.parametrize(
    'key', [
        'engine',
        'engine:threads',
        'engine:photons',
    ]
)
def test_cfg_is_invalid(key, cfg, schema, types_and_values):
    def check_ref_subschema(subschema: dict) -> dict:
        if REF_KEY_STR in subschema:
            return json.load(open(subschema[REF_KEY_STR].split(':')[-1]))
        return subschema

    # split key
    keys = key.split(':')

    # get to the right config node
    subcfg = cfg
    subschema = schema
    for subkey in keys[:-1]:
        assert subkey in subcfg
        subcfg = subcfg[subkey]

        subschema = check_ref_subschema(subschema)
        assert PROPERTIES_KEY_STR in subschema
        assert subkey in subschema[PROPERTIES_KEY_STR]
        subschema = subschema[PROPERTIES_KEY_STR][subkey]

    # check that excessive key makes cfg invalid
    assert EXCESSIVE_KEY not in subcfg
    subcfg[EXCESSIVE_KEY] = EXCESSIVE_KEY
    with pytest.raises(jsonschema.exceptions.ValidationError) as exc_info:
        jsonschema.validate(instance=cfg, schema=schema)
    assert str(exc_info.value).startswith(f"Additional properties are not allowed ('{EXCESSIVE_KEY}' was unexpected)")
    del subcfg[EXCESSIVE_KEY]

    # the key that we're currently checking
    current_key = keys[-1]

    # insert subschema is not still done
    subschema = check_ref_subschema(subschema)
    subschema[PROPERTIES_KEY_STR][current_key] = check_ref_subschema(subschema[PROPERTIES_KEY_STR][current_key])

    # check if it is an integer field and has a minimum value
    if MIN_KEY_STR in subschema[PROPERTIES_KEY_STR][current_key]:
        old_value = subcfg[current_key]
        min_value = subschema[PROPERTIES_KEY_STR][current_key][MIN_KEY_STR]
        subcfg[current_key] = min_value - 1
        with pytest.raises(jsonschema.exceptions.ValidationError) as exc_info:
            jsonschema.validate(instance=cfg, schema=schema)
        assert str(exc_info.value).startswith(f"{subcfg[current_key]} is less than the minimum of {min_value}")
        subcfg[current_key] = old_value

    # change type of the field for validation
    old_value = subcfg[current_key]
    old_type = subschema[PROPERTIES_KEY_STR][current_key][TYPE_KEY_STR]
    for value_type, value in types_and_values:
        if old_type != subschema[PROPERTIES_KEY_STR][current_key][TYPE_KEY_STR]:
            subschema[PROPERTIES_KEY_STR][current_key][TYPE_KEY_STR] = value_type
            subcfg[current_key] = value
            with pytest.raises(jsonschema.exceptions.ValidationError) as exc_info:
                jsonschema.validate(instance=cfg, schema=schema)
            assert str(exc_info.value).startswith(f"'{value}' is not of type '{old_type}'")
    subcfg[current_key] = old_value
    subschema[PROPERTIES_KEY_STR][current_key][TYPE_KEY_STR] = old_type

    # check that the key is needed
    assert current_key in subcfg
    del subcfg[current_key]
    with pytest.raises(jsonschema.exceptions.ValidationError) as exc_info:
        jsonschema.validate(instance=cfg, schema=schema)
    assert str(exc_info.value).startswith(f"'{current_key}' is a required property")
