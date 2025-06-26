FROM python:3.5.10-slim

RUN apt-get update -y && apt-get install -y wget


RUN wget https://artifactory.kinovaapps.com:443/artifactory/generic-public/kortex/API/2.6.0/kortex_api-2.6.0.post3-py3-none-any.whl && \
	 python3 -m pip install ./kortex_api-2.6.0.post3-py3-none-any.whl


WORKDIR /home/wato_research

RUN wget https://github.com/Kinovarobotics/Kinova-kortex2_Gen3_G3L/raw/refs/heads/master/api_python/examples/000-Getting_Started/01-api_creation.py && \
	wget https://raw.githubusercontent.com/Kinovarobotics/Kinova-kortex2_Gen3_G3L/refs/heads/master/api_python/examples/utilities.py

COPY /src/server /home/wato_research

CMD /bin/bash
