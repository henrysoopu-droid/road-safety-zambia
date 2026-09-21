from flask_wtf import FlaskForm
from flask_wtf.file import FileAllowed, FileField, FileRequired
from wtforms import IntegerField, PasswordField, SelectField, StringField, SubmitField, TextAreaField
from wtforms.validators import DataRequired, EqualTo, Length, NumberRange, Optional

ALLOWED_VIDEO_EXTENSIONS = ("mp4", "avi", "mov", "mkv", "webm")


class LoginForm(FlaskForm):
    username = StringField(
        "Username",
        validators=[DataRequired(), Length(min=3, max=80)],
    )
    password = PasswordField(
        "Password",
        validators=[DataRequired(), Length(min=8, max=128)],
    )
    submit = SubmitField("Sign in")


class FirstAdminForm(FlaskForm):
    username = StringField(
        "Username",
        validators=[DataRequired(), Length(min=3, max=80)],
    )
    password = PasswordField(
        "Password",
        validators=[DataRequired(), Length(min=8, max=128)],
    )
    confirm_password = PasswordField(
        "Confirm password",
        validators=[
            DataRequired(),
            EqualTo("password", message="Passwords must match."),
        ],
    )
    submit = SubmitField("Create admin account")


class UserCreateForm(FlaskForm):
    username = StringField(
        "Username",
        validators=[DataRequired(), Length(min=3, max=80)],
    )
    password = PasswordField(
        "Password",
        validators=[DataRequired(), Length(min=8, max=128)],
    )
    role = SelectField(
        "Role",
        choices=[("admin", "Admin"), ("analyst", "Analyst")],
        validators=[DataRequired()],
    )
    submit = SubmitField("Create user")


class VideoUploadForm(FlaskForm):
    title = StringField(
        "Video title",
        validators=[DataRequired(), Length(min=3, max=160)],
    )
    location = StringField(
        "Road/location",
        validators=[DataRequired(), Length(min=2, max=160)],
    )
    description = TextAreaField(
        "Description",
        validators=[Optional(), Length(max=2000)],
    )
    video = FileField(
        "Video file",
        validators=[
            FileRequired(message="Please choose a video file."),
            FileAllowed(
                ALLOWED_VIDEO_EXTENSIONS,
                "Unsupported video format. Use mp4, avi, mov, mkv, or webm.",
            ),
        ],
    )
    submit = SubmitField("Upload")


class FrameExtractionForm(FlaskForm):
    interval_seconds = IntegerField(
        "Extract every (seconds)",
        default=5,
        validators=[DataRequired(), NumberRange(min=1, max=3600)],
    )
    submit = SubmitField("Extract Frames")
