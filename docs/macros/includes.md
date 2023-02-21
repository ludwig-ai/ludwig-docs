{%- macro render_fields(fields, details=None) -%}
{%- for name, field in fields.items() -%}
    {{- render_field(name, field, details) }}
{%+ endfor -%}
{%- endmacro %}


{%- macro render_yaml(schema, parent) -%}
```yaml
{{ parent }}:
    {% for line in schema_class_to_yaml(schema).split("\n") %}
    {{- line }}
    {% endfor %}
```
{%- endmacro %}