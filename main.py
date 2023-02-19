from ludwig.schema.trainer import ECDTrainerConfig, GBMTrainerConfig

def define_env(env):
    "Hook function"

    @env.macro
    def trainer_ecd_params():
        schema = ECDTrainerConfig.get_class_schema()()
        return schema.fields
    
    @env.macro
    def render_field(name, field):
        return f"`{ name }`\n:   default: `{ field.dump_default }`\n:   { field.metadata['description'] }\n"