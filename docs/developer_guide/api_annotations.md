It's exciting to see the creative ways Ludwig has been integrated into various data science workflows and products.

To better support the engineers and scientists who use Ludwig as a platform, Ludwig has stability guarantees and expectations defined within the codebase. To make API stability clear in code, we’ve adopted the python decorators below. You can find their python implementations in this [module](https://github.com/ludwig-ai/ludwig/blob/master/ludwig/api_annotations.py) within the Ludwig codebase.

# PublicAPI

Public APIs are classes and functions exposed to end users of Ludwig.  There are two types of PublicAPI, distinguished by the stability argument. If stability is not specified, it should be assumed that `stability=”stable”`.

## PublicAPI(stability=”stable”)

A stable PublicAPI means the API is mature and will not be changed or removed in minor Ludwig releases. It may be changed in major releases, but a deprecation message will be provided in a version before the change.

## PublicAPI(stability=”experimental”)

An experimental PublicAPI is for new public features which are still in development. These APIs should be used by advanced users who are tolerant to and expect breaking changes. They will likely harden over the next 1-2 Ludwig releases and become stable PublicAPIs.

# DeveloperAPI

Developer APIs are lower-level methods explicitly exposed to advanced Ludwig users and library developers. Their interfaces may change across minor Ludwig releases.

# Deprecated

Deprecated APIs may be removed in future releases of Ludwig. Deprecated annotations will include a message with recommended alternatives, such as when a function has moved to a different import path, or the arguments of a function have changed.
