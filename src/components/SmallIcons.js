import { FontAwesomeIcon } from "@fortawesome/react-fontawesome";
import { faEnvelope, faPhone } from "@fortawesome/free-solid-svg-icons";
import { faGithub, faLinkedin } from "@fortawesome/free-brands-svg-icons";

function SmallIcons() {
    return (
        <div>
            <FontAwesomeIcon icon={faEnvelope} /> {/* Email */}
            <FontAwesomeIcon icon={faPhone} /> {/* Phone */}
            <FontAwesomeIcon icon={faGithub} /> {/* GitHub */}
            <FontAwesomeIcon icon={faLinkedin} /> {/* LinkedIn */}
        </div>
    );
}

export default SmallIcons;
