{%- macro render_fields(fields, details=None) -%}
{%- for name, field in fields.items() -%}
    {{- render_field(name, field, details) }}
{%+ endfor -%}
{%- endmacro %}