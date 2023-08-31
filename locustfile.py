from locust import HttpUser, between, task

data = dict(
    queries=['你今年几岁了啊？', '小明来自哪里？'],
    passages=['小明如今23岁了', '小明的年龄没人知道', '小明还是一个小孩子', '小红如今23岁了', '小明可不笨！', '小明很聪明的！', '小明是可靠的伙伴！', '小明喜欢的食物是甜甜圈！', '小红多大了', '小红出生在一个人鱼国家', '小红的家乡在哪里', '小红16岁的时候，离家出走了', "小小明", "小明你好"],
    topk=3,
)

class WebsiteUser(HttpUser):
    wait_time = between(1, 1.2)

    @task
    def ranks(self):
        URL = "/ranks"
        self.client.post(URL, json=data)
