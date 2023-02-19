{% macro render_fields(fields) -%}
    {% for name, field in fields.items() %}
    `{{ name }}`

    :   Default: `{{ field.dump_default }}`. {{ field.metadata["description"] }}
    {% endfor %}
{%- endmacro %}