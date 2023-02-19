{% macro render_fields(fields) -%}
    {% for name, field in fields.items() %}
    `{{ name }}`

    :   {{ render_field(field) }}
    {% endfor %}
{%- endmacro %}