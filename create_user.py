import sys
from getpass import getpass

from app import create_app
from app.extensions import db
from app.models import User

VALID_ROLES = ("admin", "analyst")


def prompt_non_empty(label):
    while True:
        value = input(label).strip()
        if value:
            return value
        print("This field is required.")


def prompt_password(label):
    try:
        if sys.stdin.isatty():
            value = getpass(label)
        else:
            value = input(label)
    except Exception:
        value = input(label)
    return value


def main():
    app = create_app()
    with app.app_context():
        db.create_all()
        print("Database:", app.config["SQLALCHEMY_DATABASE_URI"])

        username = prompt_non_empty("Username: ")
        existing = User.query.filter_by(username=username).first()
        if existing:
            print("A user with that username already exists.")
            return

        while True:
            password = prompt_password("Password: ")
            if len(password) < 8:
                print("Password must be at least 8 characters.")
                continue
            confirm = prompt_password("Confirm password: ")
            if password != confirm:
                print("Passwords do not match.")
                continue
            break

        while True:
            role = input("Role (admin/analyst): ").strip().lower()
            if role in VALID_ROLES:
                break
            print("Role must be 'admin' or 'analyst'.")

        user = User(username=username, role=role, active=True)
        user.set_password(password)
        db.session.add(user)
        db.session.commit()
        stored = User.query.filter_by(username=username).first()
        if not stored.check_password(password) or stored.password_hash == password:
            print("User was not saved correctly. Password hashing failed.")
            return
        print(f"User '{username}' created with role '{role}'.")
        print("You can now sign in at http://127.0.0.1:5000/login")


if __name__ == "__main__":
    main()
