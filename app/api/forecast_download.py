from fastapi import HTTPException
from fastapi.responses import Response
from app.core.forecast_reports import render_snapshot


def forecast_download(snapshot, lang, output_format):
    try:
        content = render_snapshot(snapshot, lang, output_format)
    except (ValueError, KeyError, TypeError):
        raise HTTPException(status_code=409, detail="The saved report is incomplete or incompatible") from None
    return Response(content, media_type="text/html" if output_format == "html" else "text/csv",
                    headers={"Content-Disposition": f'attachment; filename="forecast-{lang}.{output_format}"',
                             "Cache-Control":"no-store", "X-Content-Type-Options":"nosniff",
                             "Content-Security-Policy":"default-src 'none'; style-src 'unsafe-inline'; img-src data:; sandbox"})
