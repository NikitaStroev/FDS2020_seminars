mkdir gd3
cd gd3

git init

>> names echo Alice
>> names echo Bob
git add names && git commit -m'C0: Bob'

>> names echo Claire
git add names && git commit -m'C1: Claire'

>> names echo Diane
git add names && git commit -m'C2: Diane'

git checkout -b bugFix master~1
>> names echo Deirdre
git add names && git commit -m'C3: Deirdre'

git checkout master~2
>> names echo Catherine
git add names && git commit -m'C4: Catherine'

>> names echo Daria
git add names && git commit -m'C5: Daria'

>> names echo Eleanore
git add names && git commit -m'C6: Eleanore'
git checkout -b side

git checkout -b another side~1
>> names echo Elmo
git add names && git commit -m'C7: Elmo'

git checkout -b bak-another another
git checkout -b bak-side side
git checkout -b bak-bugFix bugFix
git checkout -b bak-master master
git checkout -b mrg-master master
git checkout master

# on each step show rebase in learngitbranching, then do the same in terminal resolving conflicts and skipping if necessary

# then address the question on dangling commits
# https://stackoverflow.com/questions/18514659/git-what-is-a-dangling-commit-blob-and-where-do-they-come-from
# git fsck, git gc, git reflog

# golden rule of git: never rebase a shared branch! https://www.daolf.com/posts/git-series-part-2/