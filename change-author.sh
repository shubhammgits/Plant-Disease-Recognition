#!/bin/sh

git filter-branch --env-filter '
# The emails to replace
OLD_EMAIL_1="94466635+kumarvivek9088@users.noreply.github.com"
OLD_EMAIL_2="kumarvivek9088@gmail.com"

# Your correct information
CORRECT_NAME="shubhammgits"
CORRECT_EMAIL="shubhamm18.work@gmail.com"

# The logic to replace the committer email
if [ "$GIT_COMMITTER_EMAIL" = "$OLD_EMAIL_1" ] || [ "$GIT_COMMITTER_EMAIL" = "$OLD_EMAIL_2" ]
then
    export GIT_COMMITTER_NAME="$CORRECT_NAME"
    export GIT_COMMITTER_EMAIL="$CORRECT_EMAIL"
fi

# The logic to replace the author email
if [ "$GIT_AUTHOR_EMAIL" = "$OLD_EMAIL_1" ] || [ "$GIT_AUTHOR_EMAIL" = "$OLD_EMAIL_2" ]
then
    export GIT_AUTHOR_NAME="$CORRECT_NAME"
    export GIT_AUTHOR_EMAIL="$CORRECT_EMAIL"
fi
' --tag-name-filter cat -- --branches --tags