FROM python:3.9.21-slim

WORKDIR /home/wato_research

COPY . /home/wato_research

# Install dependencies
RUN python3.9 -m pip install -r requirements.txt

CMD /bin/bash
