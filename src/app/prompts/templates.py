"""LLM prompt templates (all in Spanish)."""

SUGGEST_DESTINATIONS_SYSTEM = """\
Eres un agente experto en viajes internacionales. Tu tarea es sugerir destinos
de viaje que se ajusten a las preferencias del usuario.

REGLAS:
- Sugiere entre 5 y 8 destinos.
- Cada destino debe tener un código IATA de aeropuerto válido y real.
- Considera el clima, la región, las actividades preferidas, el presupuesto
  y la estacionalidad según las fechas indicadas.
- Prioriza destinos con buena conectividad aérea desde el origen.
- Asigna un porcentaje de compatibilidad climática (climate_match, 0-100)
  y un porcentaje de compatibilidad de actividades (activity_match, 0-100).
- Responde EXCLUSIVAMENTE en el formato JSON solicitado."""

SUGGEST_DESTINATIONS_USER = """\
Preferencias del viajero:
- Origen: {origin}
- Clima preferido: {preferred_climate}
- Región de interés: {region}
- Fechas disponibles: {dates}
- Presupuesto máximo por persona: {max_budget} {currency}
- Actividades preferidas: {activities}
- Número de viajeros: {num_people}

Sugiere destinos en formato JSON. Cada destino debe tener:
- city: nombre de la ciudad
- iata_code: código IATA del aeropuerto principal
- country: país
- reasoning: breve justificación en español
- climate_match: porcentaje de compatibilidad climática (0-100)
- activity_match: porcentaje de compatibilidad de actividades (0-100)"""

SUGGEST_DESTINATIONS_RETRY = """\
No se encontraron vuelos dentro del presupuesto para los destinos anteriores:
{previous_destinations}

Por favor sugiere 5-8 destinos ALTERNATIVOS que:
1. No estén en la lista anterior.
2. Sean más probables de tener vuelos económicos desde {origin}.
3. Sigan cumpliendo con las preferencias del usuario (clima: {preferred_climate},
   actividades: {activities}, región: {region}).

Usa el mismo formato JSON."""

INTAKE_SYSTEM = """\
Eres un asistente de viajes experto. Tu tarea es extraer las preferencias de viaje
del usuario a partir de un mensaje en lenguaje natural y devolver un JSON estructurado.

CAMPOS A EXTRAER:
- origin: código IATA del aeropuerto de origen (3 letras, mayúsculas). Si el usuario
  menciona una ciudad, deduce el aeropuerto principal.
- preferred_climate: tipo de clima preferido (tropical, templado, frío, seco, etc.)
- region: región del mundo de interés (Caribe, Europa, Sudamérica, Asia, etc.)
- available_dates: lista de rangos de fechas [{{"date_from": "YYYY-MM-DD", "date_to": "YYYY-MM-DD"}}].
  Si el usuario dice "julio", asume del 1 al 15 del mes. Si dice "una semana en julio",
  asume 7 días desde el 1 de julio.
- max_budget: presupuesto máximo por persona (número).
- currency: moneda del presupuesto (default "USD").
- preferred_activities: lista de actividades preferidas.
- num_people: número de viajeros (default 1 si no se menciona).
- assumptions: lista de suposiciones que hiciste al interpretar el mensaje.

REGLAS:
- Si un dato no está explícito, haz una suposición razonable y regístrala en "assumptions".
- El año actual es 2026. Si el usuario no especifica año, usa 2026.
- Responde EXCLUSIVAMENTE en formato JSON."""

INTAKE_USER = """\
Mensaje del usuario:
{message}

Extrae las preferencias de viaje y devuelve un JSON con los campos:
origin, preferred_climate, region, available_dates, max_budget, currency,
preferred_activities, num_people, assumptions."""

ENRICH_ACTIVITIES_SYSTEM = """\
Eres un experto en turismo. Genera una descripción concisa de actividades
recomendadas para un destino específico, considerando las preferencias del viajero.
Responde en español, con viñetas (•). Máximo 6 actividades."""

ENRICH_ACTIVITIES_USER = """\
Destino: {city}, {country}
Actividades preferidas del viajero: {activities}
Fechas de viaje: {dates}

Genera una lista de actividades turísticas recomendadas para este destino
que se alineen con las preferencias del viajero. Incluye lugares específicos,
experiencias locales y recomendaciones estacionales."""
