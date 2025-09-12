
"""
Module to expose more detailed version info for the installed `scipy`
"""
version = "1.11.0"
full_version = version
short_version = version.split('.dev')[0]
git_revision = "1b0b3e2a0b0c8611287f4677f21356b704495665"
release = 'dev' not in version and '+' not in version

if not release:
    version = full_version
