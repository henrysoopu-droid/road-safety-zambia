from functools import wraps

from flask import abort, Blueprint, flash, redirect, render_template, request, url_for
from flask_login import current_user, login_required, login_user, logout_user

from app.extensions import db
from app.forms import FirstAdminForm, LoginForm, UserCreateForm
from app.models import User

auth_bp = Blueprint("auth", __name__)


def admin_required(view):
    @wraps(view)
    @login_required
    def wrapped(*args, **kwargs):
        if not current_user.is_admin():
            abort(403)
        return view(*args, **kwargs)

    return wrapped


def user_count():
    return User.query.count()


@auth_bp.route("/setup", methods=["GET", "POST"])
def setup():
    if user_count() > 0:
        flash("An administrator account already exists. Please sign in.", "info")
        return redirect(url_for("auth.login"))

    form = FirstAdminForm()
    if form.validate_on_submit():
        username = form.username.data.strip()
        if User.query.filter_by(username=username).first():
            flash("A user with that username already exists.", "error")
            return render_template("setup.html", form=form)

        user = User(username=username, role="admin", active=True)
        user.set_password(form.password.data)
        db.session.add(user)
        db.session.commit()
        flash("Admin account created. You can now sign in.", "info")
        return redirect(url_for("auth.login"))

    return render_template("setup.html", form=form)


@auth_bp.route("/login", methods=["GET", "POST"])
def login():
    if current_user.is_authenticated:
        return redirect(url_for("main.dashboard"))

    if user_count() == 0:
        return redirect(url_for("auth.setup"))

    form = LoginForm()
    if form.validate_on_submit():
        user = User.query.filter_by(username=form.username.data.strip()).first()
        if user and user.active and user.check_password(form.password.data):
            login_user(user)
            next_page = request.args.get("next")
            if next_page and next_page.startswith("/"):
                return redirect(next_page)
            endpoint = "main.admin_dashboard" if user.is_admin() else "main.analyst_dashboard"
            return redirect(url_for(endpoint))
        flash("Invalid username or password.", "error")

    return render_template("login.html", form=form)


@auth_bp.route("/users", methods=["GET", "POST"])
@admin_required
def users():
    form = UserCreateForm()
    if form.validate_on_submit():
        username = form.username.data.strip()
        if User.query.filter_by(username=username).first():
            flash("A user with that username already exists.", "error")
        else:
            user = User(username=username, role=form.role.data, active=True)
            user.set_password(form.password.data)
            db.session.add(user)
            db.session.commit()
            flash(f"{username} was created as {user.role}.", "success")
            return redirect(url_for("auth.users"))

    users = User.query.order_by(User.username.asc()).all()
    return render_template("users.html", form=form, users=users)


@auth_bp.route("/users/<int:user_id>/toggle", methods=["POST"])
@admin_required
def toggle_user(user_id):
    user = db.session.get(User, user_id)
    if user is None:
        abort(404)
    if user.id == current_user.id:
        flash("You cannot disable your own account.", "error")
    else:
        user.active = not user.active
        db.session.commit()
        state = "enabled" if user.active else "disabled"
        flash(f"{user.username} is now {state}.", "success")
    return redirect(url_for("auth.users"))


@auth_bp.route("/logout", methods=["POST"])
@login_required
def logout():
    logout_user()
    flash("You have been signed out.", "info")
    return redirect(url_for("auth.login"))
