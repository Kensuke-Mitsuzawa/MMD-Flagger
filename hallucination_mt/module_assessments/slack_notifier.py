import typing as ty
import  yaml
import  psutil
import  argparse
import pathlib
from slack_sdk.webhook import WebhookClient
from  logzero  import  logger
from time import sleep



def send_message(webhook: WebhookClient, 
                  message: str,
                  user_name: ty.Optional[str] = None):
    # p_obj = psutil.Process(pid=process_id)
    # with p_obj.oneshot():
    #     # p_name = p_obj.name()
    #     p_name = p_obj
        
    # is_process_exist = psutil.pid_exists(process_id)
    user_name_code = '' if user_name is None else f'@{user_name} '
    
    message += user_name_code
    logger.info(message)
    response = webhook.send(text=message)
    
    assert response.status_code == 200
    assert response.body == "ok"


def message_monitor(webhook: WebhookClient, process_id: int, user_name: str = None, label: str = None):
    is_process_exist = psutil.pid_exists(process_id)
    
    if is_process_exist:
        p_obj = psutil.Process(pid=process_id)
        with p_obj.oneshot():
            # p_name = p_obj.name()
            p_name = str(p_obj)    
    
        message = f'[Running] p_name={is_process_exist} process_name={p_name} label={label}'
        logger.info(message)
        response = webhook.send(text=message)
        assert response.status_code == 200
        assert response.body == "ok"
        return True
    else:
        user_name_code = '' if user_name is None else f'@{user_name} '
        message = user_name_code + f'[END] it has gone. p_name={is_process_exist} process_name={p_name}'
        logger.error(message)
        response = webhook.send(text=message)
        assert response.status_code == 200
        assert response.body == "ok"
        
        raise Exception()
        