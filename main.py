import json
import yaml
from marshmallow import missing

# Force populate combiner registry:
from ludwig.constants import MODEL_ECD
from ludwig.schema.combiners.utils import get_combiner_registry
from ludwig.schema.decoders.utils import get_decoder_cls
from ludwig.schema.encoders.text_encoders import HFEncoderConfig
from ludwig.schema.encoders.utils import get_encoder_cls, get_encoder_classes
from ludwig.schema.features.augmentation.utils import get_augmentation_cls
from ludwig.schema.features.preprocessing.utils import preprocessing_registry
from ludwig.schema.features.utils import get_input_feature_cls, get_output_feature_cls
from ludwig.schema.features.loss import get_loss_schema_registry, get_loss_classes
from ludwig.schema.llms.generation import LLMGenerationConfig
from ludwig.schema.llms.model_parameters import ModelParametersConfig, RoPEScalingConfig
from ludwig.schema.llms.peft import adapter_registry
from ludwig.schema.llms.prompt import PromptConfig, RetrievalConfig
from ludwig.schema.llms.quantization import QuantizationConfig
from ludwig.schema.model_config import ModelConfig
from ludwig.schema.model_types import base
from ludwig.schema.optimizers import optimizer_registry
from ludwig.schema.preprocessing import PreprocessingConfig
from ludwig.schema.split import get_split_cls
from ludwig.schema.trainer import get_llm_trainer_cls, trainer_schema_registry


# Monkey patch the jsonschema check is it's unnedded and leads to inspect errors
base.check_schema = lambda x: None


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
    return json.dumps(v).lstrip('"').rstrip('"')


def is_internal(field):
    param_meta = field.metadata.get("parameter_metadata", {})
    if param_meta and param_meta.get("internal_only"):
        return True
    return False


def expected_impact(field):
    param_meta = field.metadata.get("parameter_metadata", {})
    if not param_meta:
        return 0
    return param_meta.get("expected_impact", 0)


def field_sort_order(name, field):
    # These fields should come at the top
    if name == "name":
        return -200
    if name == "type":
        return -100
    if name == "column":
        return -99

    return -expected_impact(field)


def sort_fields(fields_dict):
    return {
        k: v for k, v in sorted(fields_dict.items(), key=lambda x: field_sort_order(*x))
    }


def define_env(env):
    @env.macro
    def get_feature_preprocessing_schema(type: str):
        return preprocessing_registry[type]

    @env.macro
    def get_augmentation_schema(feature: str, type: str):
        return get_augmentation_cls(feature, type)

    @env.macro
    def get_input_feature_schema(type: str):
        return get_input_feature_cls(type)

    @env.macro
    def get_output_feature_schema(type: str):
        return get_output_feature_cls(type)

    @env.macro
    def get_encoder_schema(feature: str, type: str):
        return get_encoder_cls(MODEL_ECD, feature, type)

    @env.macro
    def get_decoder_schema(feature: str, type: str):
        return get_decoder_cls(MODEL_ECD, feature, type)

    @env.macro
    def get_split_schema(type: str):
        return get_split_cls(type)

    @env.macro
    def get_preprocessing_schema():
        return PreprocessingConfig

    @env.macro
    def get_loss_schema(name: str):
        return get_loss_schema_registry()[name]

    @env.macro
    def get_loss_schemas(feature: str):
        return get_loss_classes(feature).values()

    @env.macro
    def get_combiner_schema(type: str):
        return get_combiner_registry()[type]

    @env.macro
    def get_trainer_schema(model_tyoe: str):
        if model_tyoe == "llm":
            return get_llm_trainer_cls("finetune")
        return trainer_schema_registry[model_tyoe]

    @env.macro
    def get_prompt_schema():
        return PromptConfig

    @env.macro
    def get_retrieval_schema():
        return RetrievalConfig

    @env.macro
    def get_adapter_schemas():
        return [v for v in adapter_registry.values()]

    @env.macro
    def get_quantization_schema():
        return QuantizationConfig

    @env.macro
    def get_model_parameters_schema():
        return ModelParametersConfig

    @env.macro
    def get_rope_scaling_schema():
        return RoPEScalingConfig

    @env.macro
    def get_generation_schema():
        return LLMGenerationConfig

    @env.macro
    def get_optimizer_schemas():
        return [v[1] for v in optimizer_registry.values()]

    @env.macro
    def get_encoder_schemas(feature: str):
        return get_encoder_classes(feature)

    @env.macro
    def get_hf_text_encoder_schemas():
        # Sort encoders alphabetically, but put AutoTransformer first
        return sorted(
            [
                s
                for s in get_encoder_classes(MODEL_ECD, "text").values()
                if issubclass(s, HFEncoderConfig)
            ],
            key=lambda s: s.type.lower() if s.type != "auto_transformer" else "",
        )

    @env.macro
    def schema_class_long_description(cls):
        return cls.get_class_schema()().fields["type"].metadata["description"]

    @env.macro
    def schema_class_to_yaml(cls, sort_by_impact=True, exclude=None, updates=None):
        updates = updates or {}

        schema = cls.get_class_schema()()
        internal_fields = {n for n, f in schema.fields.items() if is_internal(f)}
        d = {
            k: v
            for k, v in cls(**updates).to_dict().items()
            if k not in internal_fields and k
        }

        if sort_by_impact:
            sorted_fields = flatten(sort_fields(schema.fields))
            d = {k: d[k] for k in sorted_fields.keys() if k in d}

        exclude = exclude or []
        d = {k: v for k, v in d.items() if k not in exclude}

        d.update(updates)

        return yaml.safe_dump(d, indent=4, sort_keys=False)

    @env.macro
    def schema_class_to_fields(cls, exclude=None):
        exclude = exclude or []
        schema = cls.get_class_schema()()
        d = flatten(sort_fields(schema.fields))
        return {k: v for k, v in d.items() if k not in exclude}

    @env.macro
    def render_field(name, field, details):
        if is_internal(field):
            return ""

        has_default = True
        default_value = field.dump_default
        if isinstance(default_value, dict):
            if "type" in default_value:
                default_value = {"type": default_value["type"]}
            else:
                has_default = False

        default_str = ""
        if has_default:
            if default_value == missing:
                default_value = None
            default_str = f"(default: `{dump_value(default_value)}`)"

        impact = expected_impact(field)
        impact_badge = ""
        if impact == 3:
            impact_badge = (
                ' :octicons-bookmark-fill-24:{ title="High impact parameter" }'
            )

        s = f"- **`{ name }`** {default_str}{impact_badge}: { field.metadata['description'] }"
        if field.validate is not None and hasattr(field.validate, "choices"):
            options = ", ".join(
                [f"`{dump_value(opt)}`" for opt in field.validate.choices]
            )
            s += f" Options: {options}."

        if details is not None and name in details:
            s += f" {details[name]}"

        return s

    @env.macro
    def render_config(config):
        d = ModelConfig.from_dict(config).to_dict()
        return yaml.safe_dump(d, indent=4, sort_keys=False)

    @env.macro
    def merge_dicts(d1, d2):
        return {**d1, **d2}
