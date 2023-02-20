import json
import yaml

# Force populate combiner registry:
import ludwig.combiners.combiners  # noqa: F401
from ludwig.schema.combiners.utils import get_combiner_registry
from ludwig.schema.trainer import trainer_schema_registry
from ludwig.schema.optimizers import optimizer_registry


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

    return o_dict


def dump_value(v):
    return json.dumps(v).lstrip('\"').rstrip('\"')


def define_env(env):
    @env.macro
    def get_combiner_schema(type: str):
        return get_combiner_registry()[type].get_schema_cls()
    
    @env.macro
    def get_trainer_schema(model_tyoe: str):
        return trainer_schema_registry[model_tyoe]
    
    @env.macro
    def get_optimizer_schemas():
        return [v[1] for v in optimizer_registry.values()]
    
    @env.macro
    def schema_class_to_yaml(cls):
        return yaml.safe_dump(cls().to_dict(), indent=4, sort_keys=False)
    
    @env.macro
    def schema_class_to_fields(cls, exclude=None):
        exclude = exclude or []
        schema = cls.get_class_schema()()
        d = flatten(schema.fields)
        return {
            k: v for k, v in d.items() if k not in exclude
        }
    
    @env.macro
    def render_field(name, field, details):
        has_default = True
        default_value = field.dump_default
        if isinstance(default_value, dict):
            if "type" in default_value:
                default_value = {"type": default_value["type"]}
            else:
                has_default = False

        default_str = ""
        if has_default:
            default_str = f"(default: `{dump_value(default_value)}`)"
        
        s = f"- **`{ name }`** {default_str}: { field.metadata['description'] }"
        if field.validate is not None and hasattr(field.validate, "choices"):
            options = ", ".join([f"`{dump_value(opt)}`" for opt in field.validate.choices])
            s += f" Options: {options}."

        if details is not None and name in details:
            s += f" {details[name]}"

        return s
    
    @env.macro
    def merge_dicts(d1, d2):
        return {**d1, **d2}