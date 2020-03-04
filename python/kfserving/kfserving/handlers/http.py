import tornado.web
import json
from typing import Dict, Any
from http import HTTPStatus
from kfserving.kfmodel import KFModel


class HTTPHandler(tornado.web.RequestHandler):
    def initialize(self, models: Dict[str, KFModel]):
        self.models = models # pylint:disable=attribute-defined-outside-init

    def get_model(self, name: str):
        if name not in self.models:
            raise tornado.web.HTTPError(
                status_code=HTTPStatus.NOT_FOUND,
                reason="Model with name %s does not exist." % name
            )
        model = self.models[name]
        if not model.ready:
            model.load()
        return model

    def validate(self, request):
        if "instances" in request and not isinstance(request["instances"], list):
            raise tornado.web.HTTPError(
                status_code=HTTPStatus.BAD_REQUEST,
                reason="Expected \"instances\" to be a list"
            )
        return request

    def write_error(self, status_code: int, **kwargs: Any) -> None:
        self.set_header("Content-Type", "application/json")
        self.finish(
            {"error": self._reason}
        )


class PredictHandler(HTTPHandler):
    def post(self, name: str):
        model = self.get_model(name)
        try:
            body = json.loads(self.request.body)
        except json.decoder.JSONDecodeError as e:
            raise tornado.web.HTTPError(
                status_code=HTTPStatus.BAD_REQUEST,
                reason="Unrecognized request format: %s" % e
            )
        request = model.preprocess(body)
        request = self.validate(request)
        try:
            response = model.predict(request)
        except ValueError as e:
            raise tornado.web.HTTPError(
                status_code=HTTPStatus.BAD_REQUEST,
                reason="Wrong input provided: %s" % e,
                log_message=str(e)
            )
        except Exception:
            raise tornado.web.HTTPError(
                status_code=HTTPStatus.BAD_REQUEST,
                reason="Something went wrong, probably due to bad input.",
                log_message=str(e)
            )

        response = model.postprocess(response)
        self.write(response)


class ExplainHandler(HTTPHandler):
    def post(self, name: str):
        model = self.get_model(name)
        try:
            body = json.loads(self.request.body)
        except json.decoder.JSONDecodeError as e:
            raise tornado.web.HTTPError(
                status_code=HTTPStatus.BAD_REQUEST,
                reason="Unrecognized request format: %s" % e
            )
        request = model.preprocess(body)
        request = self.validate(request)
        response = model.explain(request)
        response = model.postprocess(response)
        self.write(response)
