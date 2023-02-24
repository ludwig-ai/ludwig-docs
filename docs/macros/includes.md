{%- macro render_fields(fields, details=None) -%}
{%- for name, field in fields.items() -%}
    {{- render_field(name, field, details) }}
{%+ endfor -%}
{%- endmacro %}

{%- macro render_yaml(schema, parent=None, sort_by_impact=True, exclude=None, updates=None) -%}
{%- if parent -%}

```yaml
{{ parent }}:
    {% for line in schema_class_to_yaml(schema, sort_by_impact=sort_by_impact, exclude=exclude, updates=updates).split("\n") %}
    {{- line }}
    {% endfor %}
```

{%- else -%}

```yaml
{% for line in schema_class_to_yaml(schema, sort_by_impact=sort_by_impact, exclude=exclude, updates=updates).split("\n") %}
{{- line }}
{% endfor %}
```

{%- endif -%}
{%- endmacro %}
