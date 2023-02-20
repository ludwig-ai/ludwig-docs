{%- macro render_fields(fields) -%}
{%- for name, field in fields.items() -%}
    {{- render_field(name, field) }}
{%+ endfor -%}
{%- endmacro %}