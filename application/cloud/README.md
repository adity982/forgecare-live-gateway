# ForgeCare on Streamlit Community Cloud

This entrypoint runs the ForgeCare interface with local operations and knowledge
services in the same Cloud container. The code is shared with the laptop app.

Deploy repository `adity982/forgecare-ai`, branch `main`, main file path
`cloud/streamlit_app.py`. Select Python 3.14 in Advanced settings. Add a secret
named `CLOUD_ADMIN_PASSWORD` with a unique password of at least 10 characters.
Optionally add `GROQ_API_KEY` through Cloud secrets for generated answers. Never
commit either value. The cloud administrator username is `admin`.

Official Apple and manufacturing sources are indexed in the background after
first startup. They may take a minute to appear. The iPhone battery-condition
and front-display questions work without Groq once the relevant pages have been
indexed. The trained crack-screen check uses the included research model, but
its public-data evaluation is not an iPhone-specific validation.

This is an **ephemeral demonstration**. Community Cloud does not guarantee
persistence of local files, so captures, reviews, users, and cases can disappear
after restart or redeployment. Keep real records and six-hour federation on the
prepared laptop. A durable hosted workflow needs an external database and image
store before relying on it for inspections.
