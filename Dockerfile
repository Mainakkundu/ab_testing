FROM python:3-onbuild
RUN apt-get update && \
    apt-get -yq install gcc

COPY . /root/ab_testing
WORKDIR /root/ab_testing
COPY requirements.txt requirements.txt

RUN chmod +x boot.sh

EXPOSE 12500
CMD ["/root/ab_testing/boot.sh"]
