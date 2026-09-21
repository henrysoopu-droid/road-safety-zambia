from flask import Blueprint, Response, current_app, redirect, render_template, url_for
from flask_login import current_user, login_required

main_bp = Blueprint("main", __name__)


@main_bp.route("/healthz")
def healthz():
    return {"status": "ok"}, 200


@main_bp.route("/robots.txt")
def robots():
    sitemap_url = f"{current_app.config['SITE_URL']}/sitemap.xml"
    body = (
        "User-agent: *\n"
        "Allow: /\n"
        "Disallow: /dashboard\n"
        "Disallow: /admin/\n"
        "Disallow: /analyst/\n"
        "Disallow: /users\n"
        "Disallow: /videos\n"
        "Disallow: /setup\n"
        "Disallow: /logout\n"
        f"Sitemap: {sitemap_url}\n"
    )
    return Response(body, mimetype="text/plain")


@main_bp.route("/sitemap.xml")
def sitemap():
    public_endpoints = (
        "main.index",
        "main.about",
        "main.signs",
        "main.markings",
        "main.results",
        "auth.login",
    )
    urls = "\n".join(
        f"  <url><loc>{current_app.config['SITE_URL']}{url_for(endpoint)}</loc></url>"
        for endpoint in public_endpoints
    )
    xml = (
        '<?xml version="1.0" encoding="UTF-8"?>\n'
        '<urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">\n'
        f"{urls}\n</urlset>\n"
    )
    return Response(xml, mimetype="application/xml")

MODULE_PAGES = {
    "signs": {
        "title": "Road Signs",
        "status": "Coming Soon",
        "summary": (
            "Future versions will detect and classify traffic signs from "
            "road footage and store detection results."
        ),
        "planned": [
            "Detect traffic signs from road footage",
            "Classify detected traffic signs",
            "Store sign detection results",
        ],
    },
    "markings": {
        "title": "Road Markings",
        "status": "Coming Soon",
        "summary": (
            "Future versions will display detected road markings, support "
            "analysis, assess condition, and store results."
        ),
        "planned": [
            "View detected road markings",
            "Analyse road markings",
            "Assess road-marking condition",
            "Store marking analysis results",
        ],
    },
    "results": {
        "title": "Analysis Results",
        "status": "Coming Soon",
        "summary": (
            "This section will later present detected road signs, detected "
            "road markings, condition information, and video or frame "
            "analysis results. No analysis data is shown in Version 1."
        ),
        "planned": [
            "Detected road signs",
            "Detected road markings",
            "Condition information",
            "Video and frame analysis results",
        ],
    },
}


@main_bp.route("/")
def index():
    if current_user.is_authenticated:
        endpoint = "main.admin_dashboard" if current_user.is_admin() else "main.analyst_dashboard"
        return redirect(url_for(endpoint))
    return render_template("home.html")


@main_bp.route("/dashboard")
@login_required
def dashboard():
    endpoint = "main.admin_dashboard" if current_user.is_admin() else "main.analyst_dashboard"
    return redirect(url_for(endpoint))


@main_bp.route("/admin/dashboard")
@login_required
def admin_dashboard():
    if not current_user.is_admin():
        from flask import abort

        abort(403)
    return render_template("dashboard.html", dashboard_role="Admin")


@main_bp.route("/analyst/dashboard")
@login_required
def analyst_dashboard():
    if not current_user.is_analyst():
        from flask import abort

        abort(403)
    return render_template("dashboard.html", dashboard_role="Analyst")


@main_bp.route("/signs")
def signs():
    return render_template("module.html", module=MODULE_PAGES["signs"])


@main_bp.route("/markings")
def markings():
    return render_template("module.html", module=MODULE_PAGES["markings"])


@main_bp.route("/results")
def results():
    return render_template("module.html", module=MODULE_PAGES["results"])


@main_bp.route("/about")
def about():
    return render_template("about.html")
