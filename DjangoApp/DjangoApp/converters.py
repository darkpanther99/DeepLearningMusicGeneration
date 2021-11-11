#source: https://stackoverflow.com/questions/61860878/how-to-send-floats-as-paramters-in-django-2-0-using-path
class FloatUrlParameterConverter:
    regex = '[0-9]+\.?[0-9]+'

    def to_python(self, value):
        return float(value)

    def to_url(self, value):
        return str(value)