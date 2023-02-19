import json
import yaml

from ludwig.schema.trainer import ECDTrainerConfig, GBMTrainerConfig


def flatten(d, prefix=""):
    o_dict = {}
    for k, v in d.items():
        key = k
        if prefix:
            key = f"{prefix}.{key}"
        o_dict[key] = v

        if v is not None and hasattr(v, "load_default"):
            default = v.load_default
            if callable(default):
                default = default()
            
            cls = type(default)
            if hasattr(cls, "get_class_schema"):
                schema = cls.get_class_schema()()
                if "type" not in schema.fields:
                    o_dict.update(flatten(schema.fields, key))
    print(o_dict)
    return o_dict

def define_env(env):
    @env.macro
    def render_trainer_ecd_defaults_yaml():
        return yaml.dump(ECDTrainerConfig().to_dict(), indent=4, sort_keys=True)
    
    @env.macro
    def render_trainer_gbm_defaults_yaml():
        return yaml.dump(GBMTrainerConfig().to_dict(), indent=4, sort_keys=True)

    @env.macro
    def trainer_ecd_params():
        schema = ECDTrainerConfig.get_class_schema()()
        return flatten(schema.fields)
    
    @env.macro
    def trainer_gbm_params():
        schema = GBMTrainerConfig.get_class_schema()()
        return flatten(schema.fields)
    
    @env.macro
    def render_field(field):
        s = f"Default: `{ json.dumps(field.dump_default) }`. { field.metadata['description'] }"
        if field.validate is not None and hasattr(field.validate, "choices"):
            options = ", ".join([f"`{json.dumps(opt)}`" for opt in field.validate.choices])
            s += f" Options: {options}."
        return s