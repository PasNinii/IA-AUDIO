from datetime import datetime

class AlarmEntity:
    def __init__(self, id: int = 0,
                       city: str = "Rennes",
                       country: str = "FR",
                       threshold: float = 0.80,
                       status: bool = True,
                       created_on = datetime.now(),
                       updated_on = datetime.now(),
                       audio_path: str = "",
                       classe: str = "Unknown",
                       model_type: str = "") -> None:
        self.id = id
        self.city = city
        self.country = country
        self.threshold = threshold
        self.status = status
        self.created_on = created_on
        self.updated_on = updated_on
        self.audio_path = audio_path
        self.classe = classe
        self.model_type = model_type

    def __repr__(self) -> str:
        return f"""*** Entity: {self.id} ***
                               {self.city}\n
                               {self.country}\n
                               {self.threshold}\n
                               {self.status}\n
                               {self.created_on}\n
                               {self.updated_on}\n
                               {self.audio_path}\n
                               {self.classe}\n"""